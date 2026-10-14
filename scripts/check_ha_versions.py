#!/usr/bin/env python3
# ruff: noqa: T201  (a CLI: its output is the report)
"""
Check that the integration and its test environment agree with Home Assistant.

Home Assistant pins shared packages for every integration (its
``package_constraints.txt``) and each core component pins its own
requirements in its ``manifest.json``. Three things drift from those pins
without anyone noticing until a user's install fails:

* ``manifest.json``'s requirements stop resolving against a new HA version's
  constraints (#735: ``langchain-openai`` needed ``openai<3``, HA 2026.10
  pinned ``openai==3.10.0``), so setup fails for everyone on release day.
* ``requirements/test.txt`` pins ``homeassistant`` and
  ``pytest-homeassistant-custom-component`` separately, but the plugin
  requires one exact HA version, so the two can only move together.
* Test pins for packages HA also pins (``openai``, ``pymicro-vad``, ...) move
  past HA's version, usually through Dependabot, so the suite runs against
  library versions no real install has.

Two modes:

``pins``
    Check the HA version pinned in ``requirements/test.txt``: the plugin
    matches it, every test pin HA also pins exactly agrees with HA, and the
    manifest resolves against its constraints. Prints a notice when a newer
    stable HA and matching plugin exist. Runs on every PR.

``latest``
    Check the newest HA release on PyPI, betas included: the manifest
    resolves against its constraints, and, when a plugin release for it
    exists, write a test requirements file pinned to that HA so the workflow
    can run the suite against it. Runs on a schedule, so a conflict shows up
    during HA's beta week instead of on release day.

Needs ``uv`` on PATH for the resolution step. Everything else is stdlib.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_REQUIREMENTS = REPO_ROOT / "requirements" / "test.txt"
MANIFEST = REPO_ROOT / "custom_components" / "home_generative_agent" / "manifest.json"

HA = "homeassistant"
PLUGIN = "pytest-homeassistant-custom-component"
PYPI = "https://pypi.org/pypi"

# How many of the newest plugin releases to search for one matching an HA
# version. The plugin releases once per HA release (betas included), so the
# match for any recent HA version is well inside this window.
_PLUGIN_SEARCH_DEPTH = 40

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*?)\s*$")
_PRE_RE = re.compile(r"(a|b|rc|dev)\d*$")


def _canonical(name: str) -> str:
    """PEP 503 normalized package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement(line: str) -> tuple[str, str] | None:
    """Return (canonical name, specifier) for a requirement line, or None."""
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    match = _REQ_RE.match(line.split(";", 1)[0])
    if not match:
        return None
    return _canonical(match.group(1)), match.group(3).replace(" ", "")


def _exact_version(spec: str) -> str | None:
    """Return the version of an ``==X`` specifier, or None for anything else."""
    if spec.startswith("==") and "," not in spec and "*" not in spec:
        return spec[2:]
    return None


def _version_key(version: str) -> tuple[tuple[int, ...], int, int]:
    """
    Sort key for HA-style versions: 2026.10.1 > 2026.10.1b2 > 2026.10.0.

    Enough for the versions this script compares (HA calver, the plugin's
    0.13.N); not a general PEP 440 implementation.
    """
    pre = _PRE_RE.search(version)
    base = version[: pre.start()] if pre else version
    nums = tuple(int(p) for p in base.split(".") if p.isdigit())
    if not pre:
        return nums, 1, 0
    digits = re.sub(r"\D", "", pre.group(0))
    return nums, 0, int(digits or 0)


def _is_prerelease(version: str) -> bool:
    return bool(_PRE_RE.search(version))


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read()


def _pypi(package: str, version: str | None = None) -> dict[str, Any]:
    path = f"{package}/{version}/json" if version else f"{package}/json"
    return json.loads(_fetch(f"{PYPI}/{path}"))


def _plugin_ha_version(plugin_version: str) -> str | None:
    """Return the exact HA version a plugin release requires."""
    for req in _pypi(PLUGIN, plugin_version)["info"].get("requires_dist") or []:
        parsed = _parse_requirement(req)
        if parsed and parsed[0] == HA:
            return _exact_version(parsed[1])
    return None


def _plugin_for(ha_version: str) -> str | None:
    """Return the newest plugin release that requires exactly ``ha_version``."""
    releases = _pypi(PLUGIN)["releases"]
    newest = sorted(
        (v for v, files in releases.items() if files), key=_version_key, reverse=True
    )
    for version in newest[:_PLUGIN_SEARCH_DEPTH]:
        if _plugin_ha_version(version) == ha_version:
            return version
    return None


def _ha_releases() -> list[str]:
    releases = _pypi(HA)["releases"]
    return sorted(
        (
            v
            for v, files in releases.items()
            if files and not any(f.get("yanked") for f in files)
        ),
        key=_version_key,
    )


class HAPins:
    """The packages one HA version pins: its constraints and its components."""

    def __init__(self, version: str) -> None:
        """Download the HA wheel for ``version`` and read its pins."""
        self.version = version
        files = _pypi(HA, version)["urls"]
        wheel = next(f for f in files if f["filename"].endswith(".whl"))
        with zipfile.ZipFile(io.BytesIO(_fetch(wheel["url"]))) as zf:
            self.constraints_text = zf.read(
                "homeassistant/package_constraints.txt"
            ).decode()
            # Component pins can differ between components; keep each one.
            self.component_pins: dict[str, set[str]] = {}
            for name in zf.namelist():
                if not re.fullmatch(
                    r"homeassistant/components/[^/]+/manifest\.json", name
                ):
                    continue
                for req in json.loads(zf.read(name)).get("requirements", []):
                    parsed = _parse_requirement(req)
                    exact = parsed and _exact_version(parsed[1])
                    if parsed and exact:
                        self.component_pins.setdefault(parsed[0], set()).add(exact)

        self.constraint_pins: dict[str, str] = {}
        for line in self.constraints_text.splitlines():
            parsed = _parse_requirement(line)
            exact = parsed and _exact_version(parsed[1])
            if parsed and exact:
                self.constraint_pins[parsed[0]] = exact

    def pinned(self, package: str) -> set[str]:
        """Every exact version HA pins ``package`` to (empty if it does not)."""
        if package in self.constraint_pins:
            # A constraint applies to every install, whatever a component asks.
            return {self.constraint_pins[package]}
        return self.component_pins.get(package, set())


def _test_pins() -> dict[str, tuple[str, str]]:
    """Canonical name -> (specifier, original line) from requirements/test.txt."""
    pins: dict[str, tuple[str, str]] = {}
    for line in TEST_REQUIREMENTS.read_text().splitlines():
        parsed = _parse_requirement(line)
        if parsed:
            pins[parsed[0]] = (parsed[1], line.strip())
    return pins


def _manifest_requirements() -> list[str]:
    return json.loads(MANIFEST.read_text())["requirements"]


def _resolve_manifest(ha: HAPins) -> str | None:
    """
    Resolve the manifest's requirements under HA's constraints with uv.

    The same thing HA does when it installs the integration. Returns uv's
    error output on failure, None on success.
    """
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp, "requirements.in")
        reqs.write_text("\n".join(_manifest_requirements()) + "\n")
        constraints = Path(tmp, "constraints.txt")
        constraints.write_text(ha.constraints_text)
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        proc = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "pip",
                "compile",
                "--quiet",
                "--no-header",
                "--python-version",
                python,
                "--constraint",
                str(constraints),
                str(reqs),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode == 0:
        return None
    return (proc.stderr or proc.stdout).strip()


def _annotate(level: str, message: str) -> None:
    """Print ``message``, as a GitHub annotation when running in Actions."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message.replace(chr(10), '%0A')}")
    else:
        print(f"{level.upper()}: {message}")


def _check_test_pins(ha: HAPins, pins: dict[str, tuple[str, str]]) -> list[str]:
    """Test pins that disagree with what HA pins for the same package."""
    problems = []
    for package, (spec, line) in sorted(pins.items()):
        if package in (HA, _canonical(PLUGIN)):
            continue
        expected = ha.pinned(package)
        if not expected:
            continue
        if _exact_version(spec) not in expected:
            want = " or ".join(f"=={v}" for v in sorted(expected, key=_version_key))
            problems.append(
                f"requirements/test.txt has '{line}', but Home Assistant "
                f"{ha.version} pins {package}{want}. Tests should run against "
                "the version real installs get."
            )
    return problems


def _cmd_pins() -> int:
    pins = _test_pins()
    ha_version = _exact_version(pins[HA][0]) if HA in pins else None
    plugin_version = (
        _exact_version(pins[_canonical(PLUGIN)][0])
        if _canonical(PLUGIN) in pins
        else None
    )
    if not ha_version or not plugin_version:
        _annotate(
            "error",
            f"requirements/test.txt must pin {HA} and {PLUGIN} exactly (==).",
        )
        return 1

    problems: list[str] = []
    plugin_ha = _plugin_ha_version(plugin_version)
    if plugin_ha != ha_version:
        problems.append(
            f"{PLUGIN}=={plugin_version} requires {HA}=={plugin_ha}, but "
            f"requirements/test.txt pins {HA}=={ha_version}. They move together: "
            f"use the plugin release built for {ha_version}"
            + (f" ({_plugin_for(ha_version)})." if _plugin_for(ha_version) else ".")
        )

    ha = HAPins(ha_version)
    problems += _check_test_pins(ha, pins)
    if error := _resolve_manifest(ha):
        problems.append(
            f"manifest.json requirements do not resolve against Home Assistant "
            f"{ha_version}'s package constraints:\n{error}"
        )

    for problem in problems:
        _annotate("error", problem)

    stable = [v for v in _ha_releases() if not _is_prerelease(v)]
    if stable and _version_key(stable[-1]) > _version_key(ha_version):
        newer_plugin = _plugin_for(stable[-1])
        if newer_plugin:
            _annotate(
                "notice",
                f"Home Assistant {stable[-1]} is out; tests run {ha_version}. "
                f"Bump requirements/test.txt to {HA}=={stable[-1]} and "
                f"{PLUGIN}=={newer_plugin} (and the pins this check names).",
            )

    if not problems:
        print(
            f"OK: {HA}=={ha_version}, {PLUGIN}=={plugin_version}, test pins "
            "and manifest requirements agree with Home Assistant."
        )
    return 1 if problems else 0


def _test_requirements_for(ha: HAPins, plugin_version: str) -> str:
    """Return requirements/test.txt with every HA-governed pin moved to ``ha``."""
    lines = []
    for line in TEST_REQUIREMENTS.read_text().splitlines():
        parsed = _parse_requirement(line)
        if parsed is None:
            lines.append(line)
        elif parsed[0] == HA:
            lines.append(f"{HA}=={ha.version}")
        elif parsed[0] == _canonical(PLUGIN):
            lines.append(f"{PLUGIN}=={plugin_version}")
        elif len(pinned := ha.pinned(parsed[0])) == 1:
            lines.append(f"{parsed[0]}=={next(iter(pinned))}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _cmd_latest(requirements_out: Path | None, constraints_out: Path | None) -> int:
    ha_version = _ha_releases()[-1]
    print(f"Newest Home Assistant on PyPI: {ha_version}")
    ha = HAPins(ha_version)

    failed = False
    if error := _resolve_manifest(ha):
        _annotate(
            "error",
            f"manifest.json requirements do not resolve against Home Assistant "
            f"{ha_version}'s package constraints, so setup will fail for every "
            f"user on that version:\n{error}",
        )
        failed = True
    else:
        print(f"OK: manifest.json resolves against {ha_version}'s constraints.")

    plugin_version = _plugin_for(ha_version)
    if constraints_out:
        constraints_out.write_text(ha.constraints_text)
    if requirements_out:
        if plugin_version is None:
            _annotate(
                "notice",
                f"No {PLUGIN} release for {ha_version} yet; skipping the test run.",
            )
        else:
            requirements_out.write_text(_test_requirements_for(ha, plugin_version))
            print(f"Test requirements for {ha_version} ({PLUGIN}=={plugin_version})")
            print(f"written to {requirements_out}")

    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as fh:
            fh.write(f"ha_version={ha_version}\n")
            fh.write(f"run_tests={'true' if plugin_version else 'false'}\n")
    return 1 if failed else 0


def main() -> int:
    """Run the check named on the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("pins", help="check the pinned HA version (PR check)")
    latest = sub.add_parser("latest", help="check the newest HA release, betas too")
    latest.add_argument(
        "--requirements-out",
        type=Path,
        help="write test requirements pinned to the newest HA here",
    )
    latest.add_argument(
        "--constraints-out",
        type=Path,
        help="write the newest HA's package_constraints.txt here",
    )
    args = parser.parse_args()
    if args.mode == "pins":
        return _cmd_pins()
    return _cmd_latest(args.requirements_out, args.constraints_out)


if __name__ == "__main__":
    sys.exit(main())

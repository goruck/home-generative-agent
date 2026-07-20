# ruff: noqa: S101
"""
Parity checks for strings.json and the translation files (Issue #494).

Guards against drift between strings.json, en.json, and the localized
translation files. Non-English files may lag behind en.json (Home Assistant
falls back to English for missing keys), but they must never contain keys
that do not exist in en.json, and placeholders must match on shared keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

COMPONENT_DIR = (
    Path(__file__).parents[3] / "custom_components" / "home_generative_agent"
)
TRANSLATIONS_DIR = COMPONENT_DIR / "translations"

_PLACEHOLDER_RE = re.compile(r"{\w+}")


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested translation dict into dotted-key/string pairs."""
    out: dict[str, str] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def _load(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fp:
        return _flatten(json.load(fp))


def _translation_files() -> list[Path]:
    files = sorted(TRANSLATIONS_DIR.glob("*.json"))
    assert files, "no translation files found"
    return files


def test_en_json_matches_strings_json() -> None:
    """translations/en.json must be an exact copy of strings.json."""
    strings = _load(COMPONENT_DIR / "strings.json")
    en = _load(TRANSLATIONS_DIR / "en.json")
    assert en == strings


def test_cs_json_has_full_key_parity_with_en() -> None:
    """cs.json is a complete translation: exact key parity with en.json."""
    en = _load(TRANSLATIONS_DIR / "en.json")
    cs = _load(TRANSLATIONS_DIR / "cs.json")
    missing = set(en) - set(cs)
    extra = set(cs) - set(en)
    assert not missing, f"cs.json missing keys: {sorted(missing)}"
    assert not extra, f"cs.json has unknown keys: {sorted(extra)}"


@pytest.mark.parametrize("path", _translation_files(), ids=lambda p: p.name)
def test_translation_files_have_no_unknown_keys(path: Path) -> None:
    """No translation file may contain keys absent from en.json."""
    en = _load(TRANSLATIONS_DIR / "en.json")
    translated = _load(path)
    extra = set(translated) - set(en)
    assert not extra, f"{path.name} has keys not in en.json: {sorted(extra)}"


@pytest.mark.parametrize("path", _translation_files(), ids=lambda p: p.name)
def test_translation_values_have_no_surrounding_whitespace(path: Path) -> None:
    """Hassfest rejects values with leading/trailing whitespace — enforce locally."""
    translated = _load(path)
    bad = sorted(
        key
        for key, value in translated.items()
        if isinstance(value, str) and value != value.strip()
    )
    assert not bad, f"{path.name} values with surrounding whitespace: {bad}"


@pytest.mark.parametrize("path", _translation_files(), ids=lambda p: p.name)
def test_translation_placeholders_match_en(path: Path) -> None:
    """Shared keys must use the same {placeholder} set as en.json."""
    en = _load(TRANSLATIONS_DIR / "en.json")
    translated = _load(path)
    mismatches = {
        key: (
            sorted(set(_PLACEHOLDER_RE.findall(en[key]))),
            sorted(set(_PLACEHOLDER_RE.findall(translated[key]))),
        )
        for key in set(en) & set(translated)
        if set(_PLACEHOLDER_RE.findall(en[key]))
        != set(_PLACEHOLDER_RE.findall(translated[key]))
    }
    assert not mismatches, f"{path.name} placeholder mismatches: {mismatches}"


def test_setup_mode_localization_keys_present() -> None:
    """The Issue #494 keys exist in strings.json."""
    strings = _load(COMPONENT_DIR / "strings.json")
    for subentry in ("sentinel", "feature"):
        key = f"common.{subentry}_overwrite_warning"
        assert key in strings, f"missing {key}"
        assert strings[key], f"{key} is empty"
    assert strings["selector.setup_mode.options.basic"] == "Basic setup"
    assert strings["selector.setup_mode.options.advanced"] == "Advanced setup"

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
from string import Formatter
from typing import Any

import pytest

COMPONENT_DIR = (
    Path(__file__).parents[3] / "custom_components" / "home_generative_agent"
)
TRANSLATIONS_DIR = COMPONENT_DIR / "translations"

_PLACEHOLDER_RE = re.compile(r"{\w+}")

# Hassfest requires every `{...}` in a translation value to name a valid
# Python identifier, because Home Assistant renders these through
# ``str.format``.  A literal brace must be doubled (``{{`` / ``}}``).
_IDENTIFIER_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


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


def _hassfest_validated_files() -> list[Path]:
    """Every file hassfest runs its translation checks over."""
    return [COMPONENT_DIR / "strings.json", *_translation_files()]


def _invalid_placeholders(value: str) -> list[str]:
    """
    Return the ``{...}`` fields in *value* that hassfest would reject.

    Mirrors ``str.format`` semantics via ``string.Formatter``, so doubled
    braces are correctly read as escaped literals rather than placeholders.
    A value that will not parse at all (a stray unmatched brace) is reported
    too — hassfest rejects those as well.
    """
    try:
        fields = [
            field
            for _literal, field, _spec, _conv in Formatter().parse(value)
            if field is not None
        ]
    except ValueError as err:  # unmatched or malformed brace
        return [f"<unparseable: {err}>"]
    return [f for f in fields if not _IDENTIFIER_RE.fullmatch(f)]


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


@pytest.mark.parametrize("path", _hassfest_validated_files(), ids=lambda p: p.name)
def test_translation_placeholders_are_valid_identifiers(path: Path) -> None:
    """
    Every ``{...}`` must name an identifier — literal braces must be doubled.

    Hassfest enforces this and only runs in CI, so an unescaped JSON example
    in a help string turns the whole integration invalid with a green
    ``make lint`` and a green ``make test``.  That is exactly how it reached
    the branch for PR #544: ``Format: {"rule_id": ["entity.id", ...]}`` in the
    advanced-exclusions ``data_description``, while the sibling error string
    in the same file already used the doubled-brace escape.
    """
    bad = {
        key: invalid
        for key, value in _load(path).items()
        if isinstance(value, str) and (invalid := _invalid_placeholders(value))
    }
    assert not bad, (
        f"{path.name} has braces hassfest will reject (double them to escape "
        f"a literal brace): {bad}"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The literal string that turned hassfest red on PR #544.
        ('Format: {"rule_id": ["entity.id", ...]}', ['"rule_id"']),
        ("{not-an-identifier}", ["not-an-identifier"]),
        ("{}", [""]),
        ("{0}", ["0"]),
    ],
)
def test_invalid_placeholders_rejects(value: str, expected: list[str]) -> None:
    """Pin the guard itself, so it cannot rot into always-passing."""
    assert _invalid_placeholders(value) == expected


def test_invalid_placeholders_reports_unparseable_braces() -> None:
    """A stray brace is reported rather than crashing the test run."""
    assert _invalid_placeholders("stray closing brace }")


@pytest.mark.parametrize(
    "value",
    [
        # The fixed form of the PR #544 string, matching the sibling error
        # string in the same file.
        'Format: {{"rule_id": ["entity.id", ...]}}',
        "no braces at all",
        "a real {placeholder} is fine",
        "{{escaped}} beside a real {placeholder}",
    ],
)
def test_invalid_placeholders_accepts(value: str) -> None:
    """Escaped literals and real placeholders must not be flagged."""
    assert _invalid_placeholders(value) == []


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

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

# Two engines read these strings, and they disagree about braces:
#
#   * Hassfest validates with ``str.format`` semantics — every ``{x}`` must
#     name a valid identifier, and a literal brace is doubled (``{{``).
#   * The HA *frontend* renders the same strings through ICU MessageFormat
#     (``intl-messageformat``), where ``{{`` is not an escape at all — it is
#     a malformed argument, and the field shows
#     "Translation error: MALFORMED_ARGUMENT" instead of the help text.
#
# So no brace spelling satisfies both: single braces fail hassfest, doubled
# braces fail the frontend, and ICU's own apostrophe escape (``'{'``) would
# fail hassfest in turn.  The only safe text is text with no literal braces —
# write the JSON example as prose.  Square brackets are fine in both.
_IDENTIFIER_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_BRACED_RE = re.compile(r"\{([^{}]*)\}")


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
    Return the brace usages in *value* that hassfest or the frontend rejects.

    Enforces the intersection of both engines (see the module-level note):
    a literal brace is never allowed in either spelling, and every remaining
    ``{...}`` must name a valid identifier so it is a real placeholder.
    """
    problems: list[str] = []
    if "{{" in value or "}}" in value:
        problems.append("{{ or }} — escaped for hassfest, MALFORMED_ARGUMENT in the UI")
    problems += [
        f"{{{m.group(1)}}}"
        for m in _BRACED_RE.finditer(value)
        if not _IDENTIFIER_RE.fullmatch(m.group(1))
    ]
    if value.count("{") != value.count("}"):
        problems.append("unbalanced braces")
    return problems


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
    Only real ``{identifier}`` placeholders — never a literal brace.

    Both failure modes have shipped, one after the other, on PR #544:

    * ``Format: {"rule_id": ["entity.id", ...]}`` turned hassfest red, which
      only runs in CI, so ``make lint`` and ``make test`` stayed green.
    * Doubling the braces to satisfy hassfest then rendered the field as
      "Translation error: MALFORMED_ARGUMENT" in the UI, because the frontend
      parses ICU MessageFormat, where ``{{`` is not an escape.

    Neither spelling works, so neither is allowed: write the JSON example as
    prose instead.
    """
    bad = {
        key: invalid
        for key, value in _load(path).items()
        if isinstance(value, str) and (invalid := _invalid_placeholders(value))
    }
    assert not bad, (
        f"{path.name} has brace usage one of the two engines rejects — write "
        f"the example as prose rather than literal JSON: {bad}"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The string that turned hassfest red on PR #544...
        (
            'Format: {"rule_id": ["entity.id", ...]}',
            ['{"rule_id": ["entity.id", ...]}'],
        ),
        # ...and the doubled-brace "fix" that satisfied hassfest but rendered
        # as MALFORMED_ARGUMENT in the UI. Both must be rejected.
        (
            'Format: {{"rule_id": ["entity.id", ...]}}',
            [
                "{{ or }} — escaped for hassfest, MALFORMED_ARGUMENT in the UI",
                '{"rule_id": ["entity.id", ...]}',
            ],
        ),
        ("{not-an-identifier}", ["{not-an-identifier}"]),
        ("{}", ["{}"]),
        ("{0}", ["{0}"]),
        ("stray closing brace }", ["unbalanced braces"]),
    ],
)
def test_invalid_placeholders_rejects(value: str, expected: list[str]) -> None:
    """Pin the guard itself, so it cannot rot into always-passing."""
    assert _invalid_placeholders(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        # The shipped fix: the same example written as prose. Square brackets
        # are literal in both engines, so only the braces had to go.
        'an "appliance_power_duration" key with the value ["sensor.ac_power"]',
        "no braces at all",
        "a real {placeholder} is fine",
        "two {placeholders} in {one} string",
    ],
)
def test_invalid_placeholders_accepts(value: str) -> None:
    """Real placeholders and brace-free prose must not be flagged."""
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

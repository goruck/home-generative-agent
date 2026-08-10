# ruff: noqa: S101
"""Tests for the localized critical-action PIN flow messages."""

from string import Formatter
from types import SimpleNamespace
from typing import Any

from custom_components.home_generative_agent.agent.pin_messages import (
    _DEFAULT_LANGUAGE,
    _MESSAGES,
    pin_msg,
)
from custom_components.home_generative_agent.agent.tools import (
    _validate_pin_for_action,
)

# One dummy value per placeholder name used anywhere in the tables. A KeyError
# in test_templates_format_cleanly means a language introduced a placeholder
# without adding it here (and without it existing in the English template).
_DUMMY_KWARGS: dict[str, Any] = {
    "reason": "reason",
    "calls": "calls",
    "min_len": 4,
    "max_len": 10,
    "err": "err",
}


def _fake_hass(language: str) -> Any:
    return SimpleNamespace(config=SimpleNamespace(language=language))


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def test_default_language_is_english():
    assert _DEFAULT_LANGUAGE == "en"
    assert _DEFAULT_LANGUAGE in _MESSAGES


def test_every_language_has_exactly_the_english_key_set():
    """A key present in only one language silently degrades to the raw key."""
    english_keys = set(_MESSAGES[_DEFAULT_LANGUAGE])
    for language, table in _MESSAGES.items():
        assert set(table) == english_keys, (
            f"language {language!r} keys diverge from English"
        )


def test_placeholders_match_english_per_key():
    """Translations must use the same format placeholders as the English text."""
    for language, table in _MESSAGES.items():
        for key, template in table.items():
            expected = _placeholders(_MESSAGES[_DEFAULT_LANGUAGE][key])
            assert _placeholders(template) == expected, (language, key)


def test_templates_format_cleanly():
    """Malformed braces or unknown placeholders raise at format time."""
    for table in _MESSAGES.values():
        for template in table.values():
            needed = _placeholders(template)
            formatted = template.format(**{k: _DUMMY_KWARGS[k] for k in needed})
            assert "{" not in formatted


def test_none_hass_falls_back_to_english():
    assert pin_msg(None, "pin_incorrect") == "Incorrect PIN. Action not executed."


def test_unsupported_language_falls_back_to_english():
    hass = _fake_hass("de")
    assert pin_msg(hass, "pin_incorrect") == "Incorrect PIN. Action not executed."


def test_czech_language_resolves_czech_string():
    hass = _fake_hass("cs")
    assert pin_msg(hass, "pin_incorrect") == "Nesprávný PIN. Akce nebyla provedena."


def test_region_suffix_resolves_base_language():
    hass = _fake_hass("cs-CZ")
    assert pin_msg(hass, "pin_incorrect") == "Nesprávný PIN. Akce nebyla provedena."


def test_unknown_key_returns_the_key_itself():
    assert pin_msg(None, "no_such_message_id") == "no_such_message_id"


def test_format_kwargs_are_applied():
    hass = _fake_hass("cs")
    result = pin_msg(hass, "pin_invalid_format", min_len=4, max_len=10)
    assert "4-10" in result
    assert "{" not in result


def test_validate_pin_for_action_threads_hass_language():
    """An incorrect PIN on a Czech instance produces the Czech message."""
    result = _validate_pin_for_action(
        provided_pin="123456",
        pin_hash="deadbeef",
        salt="0123456789abcdef",
        action={"attempts": 0},
        hass=_fake_hass("cs"),
    )
    assert result == "Nesprávný PIN. Akce nebyla provedena."

# ruff: noqa: S101
"""Tests for the localized Sentinel notification-chrome messages."""

from string import Formatter
from types import SimpleNamespace
from typing import Any

from custom_components.home_generative_agent.sentinel.notifier import (
    _KNOWN_TYPE_LABEL_KEYS,
)
from custom_components.home_generative_agent.sentinel.notifier_messages import (
    _DEFAULT_LANGUAGE,
    _MESSAGES,
    notif_msg,
)

# One dummy value per placeholder name used anywhere in the tables. A KeyError
# in test_templates_format_cleanly means a language introduced a placeholder
# without adding it here (and without it existing in the English template).
_DUMMY_KWARGS: dict[str, Any] = {
    "appliance": "Washer",
    "entry_name": "Front Door",
    "direction": "higher",
    "summary": "Motion while away",
    "entity": "Front Door",
    "entities": "Front Door, Back Door",
    "action_hint": "Review when convenient.",
    "hint": "Review when convenient.",
    "severity": "high",
    "count": 2,
    "plural": "s",
    "type_summary": "Motion while away",
    "sev_summary": "2 high",
    "friendly": "Motion while away",
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


def test_placeholders_are_subset_of_english_per_key():
    """
    Translations may drop placeholders but never add them.

    Czech pluralization legitimately ignores the English ``{plural}``
    suffix, but a placeholder the call site does not supply would
    KeyError at notification time.
    """
    for language, table in _MESSAGES.items():
        for key, template in table.items():
            allowed = _placeholders(_MESSAGES[_DEFAULT_LANGUAGE][key])
            assert _placeholders(template) <= allowed, (language, key)


def test_templates_format_cleanly():
    """Malformed braces or unknown placeholders raise at format time."""
    for table in _MESSAGES.values():
        for template in table.values():
            needed = _placeholders(template)
            formatted = template.format(**{k: _DUMMY_KWARGS[k] for k in needed})
            assert "{" not in formatted


def test_known_type_label_keys_all_resolve_to_english_strings():
    """
    Every type-label key must exist in the English message table.

    Otherwise notifications would show raw message ids like
    ``type_appliance_power_duration`` (the #537 pitfall class).
    """
    english = _MESSAGES[_DEFAULT_LANGUAGE]
    for anomaly_type, message_key in _KNOWN_TYPE_LABEL_KEYS.items():
        assert message_key in english, (anomaly_type, message_key)


def test_none_hass_falls_back_to_english():
    assert notif_msg(None, "severity_title_high") == "Security Alert"


def test_hass_without_config_falls_back_to_english():
    """Minimal test doubles (and any odd runtime state) must not crash."""
    bare: Any = SimpleNamespace()
    assert notif_msg(bare, "severity_title_high") == "Security Alert"


def test_unsupported_language_falls_back_to_english():
    hass = _fake_hass("de")
    assert notif_msg(hass, "severity_title_high") == "Security Alert"


def test_czech_language_resolves_czech_string():
    hass = _fake_hass("cs")
    assert notif_msg(hass, "severity_title_high") == "Bezpečnostní výstraha"


def test_region_suffix_resolves_base_language():
    hass = _fake_hass("cs-CZ")
    assert notif_msg(hass, "severity_title_high") == "Bezpečnostní výstraha"


def test_unknown_key_returns_the_key_itself():
    assert notif_msg(None, "no_such_message_id") == "no_such_message_id"


def test_format_kwargs_are_applied():
    hass = _fake_hass("cs")
    result = notif_msg(hass, "subtitle_entry_open_alarm_disarmed", entry_name="Dveře")
    assert "Dveře" in result
    assert "{" not in result


def test_extra_kwargs_are_ignored():
    """Czech drops ``{plural}``; the unused kwarg must not raise."""
    hass = _fake_hass("cs")
    result = notif_msg(hass, "batch_message", count=1, plural="s", type_summary="Pohyb")
    assert "Pohyb" in result
    assert "{" not in result


# The exact kwarg names each notifier.py call site supplies per message key.
# Every template's placeholders (in every language) must be a subset of these,
# or notif_msg would have to fall back at notification time. Keys absent here
# are rendered without kwargs and must contain no placeholders at all.
_CALL_SITE_KWARGS: dict[str, set[str]] = {
    "subtitle_appliance_finished": {"appliance"},
    "subtitle_entry_open_alarm_disarmed": {"entry_name"},
    "subtitle_power_deviation": {"appliance", "direction"},
    "subtitle_reading_deviation": {"appliance", "direction"},
    "fallback_message": {"summary", "entity", "action_hint"},
    "persistent_fallback": {"summary", "severity", "entities", "hint"},
    "batch_message": {"count", "plural", "type_summary"},
    "digest_message": {"count", "plural", "sev_summary"},
    "snooze_confirm_message": {"friendly"},
}


def test_placeholders_match_what_call_sites_supply():
    """
    Template placeholders must never exceed the call site's kwargs.

    ``_DUMMY_KWARGS``-based formatting cannot catch a template that grows a
    placeholder the real call site does not pass (e.g. adding ``{severity}``
    to ``fallback_message``); this contract test does.
    """
    for language, table in _MESSAGES.items():
        for key, template in table.items():
            supplied = _CALL_SITE_KWARGS.get(key, set())
            assert _placeholders(template) <= supplied, (language, key)


def test_bad_placeholder_degrades_instead_of_crashing(monkeypatch):
    """A typo'd placeholder in a translation must not raise on the alert path."""
    monkeypatch.setitem(
        _MESSAGES["cs"], "snooze_confirm_message", "Potlačit '{freindly}'?"
    )
    hass = _fake_hass("cs")
    result = notif_msg(hass, "snooze_confirm_message", friendly="Pohyb")
    # Falls back to the (valid) English template rather than raising.
    assert "Pohyb" in result
    assert result == _MESSAGES["en"]["snooze_confirm_message"].format(friendly="Pohyb")


def test_unknown_key_with_kwargs_is_never_formatted():
    """Data-derived ids must not be treated as format templates."""
    assert notif_msg(None, "{count}", count=9) == "{count}"

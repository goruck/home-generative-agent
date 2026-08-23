"""
Localized fixed strings for Sentinel notification chrome.

Covers titles, subtitles, type labels, and the batch/digest/snooze
summary strings -- these are the pieces of a Sentinel notification
that are *not* LLM output: severity titles ("Security Alert"), finding-type labels
("Open entry while away"), subtitle templates, the burst-batch summary,
the daily digest, and the permanent-snooze confirmation prompt. They
are built directly in ``notifier.py`` as Python string literals and
never pass through ``strings.json`` / ``translations/*.json`` (see
discussion on #523 / #531).

This module gives them the same "pick a language, fall back to
English" treatment as ``agent/pin_messages.py``, resolved from
``hass.config.language`` -- *not* ``CONF_SENTINEL_RESPONSE_LANGUAGE``,
which is a free-text LLM prompt fragment ("Czech", "please respond in
German") rather than a language code, and so is not usable as a table
key. Using ``hass.config.language`` also means these strings translate
automatically for a Czech-configured instance even if
``sentinel_response_language`` (the LLM explainer option) is left
unset.

Deliberately out of scope: the deterministic per-template mobile
message bodies (``_deterministic_mobile_message`` and friends in
``notifier.py``) stay English-only by design -- they carry exact
figures/units/entity names that a translation could blur, per the
#523 review discussion. Only the chrome around them is localized here.

Add a new language by adding a key to ``_MESSAGES`` with the same set
of message ids as ``"en"``. Missing keys in a non-English language
fall back to the English string automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "severity_title_high": "Security Alert",
        "severity_title_medium": "Home Alert",
        "severity_title_low": "Home Update",
        "type_open_entry_while_away": "Open entry while away",
        "type_open_entry_at_night": "Open entry at night",
        "type_open_any_window_at_night_while_away": "Window open at night",
        "type_motion_detected_at_night_while_away": "Motion at night while away",
        "type_motion_detected_while_away": "Motion while away",
        "type_unlocked_lock_at_night": "Door lock left unlocked",
        "type_camera_entry_unsecured": "Activity near unsecured entry",
        "type_alarm_disarmed_during_external_threat": (
            "Outdoor activity while alarm disarmed"
        ),
        "type_appliance_power_duration": "Appliance power duration",
        "subtitle_appliance_finished": "{appliance} finished",
        "subtitle_appliance_cycle_complete": "Appliance cycle complete",
        "subtitle_entry_open_alarm_disarmed": "{entry_name} open, alarm disarmed",
        "subtitle_power_deviation": "{appliance}: power {direction} than expected",
        "subtitle_reading_deviation": "{appliance}: reading {direction} than expected",
        "direction_lower": "lower",
        "direction_higher": "higher",
        "fallback_entry": "Entry",
        "fallback_sensor": "Sensor",
        "fallback_unknown_entity": "Unknown entity",
        "fallback_message": "{summary}: {entity}. {action_hint}",
        "persistent_fallback": "{summary} (severity {severity}) for {entities}. {hint}",
        "batch_title": "Home Update",
        "batch_message": "{count} home update{plural}: {type_summary}.",
        "digest_title": "Sentinel Daily Digest",
        "digest_message": (
            "Sentinel: {count} alert{plural} in the last 24 h ({sev_summary})."
        ),
        "severity_word_high": "high",
        "severity_word_medium": "medium",
        "severity_word_low": "low",
        "snooze_confirm_title": "Confirm permanent snooze",
        "snooze_confirm_message": (
            "Permanently suppress '{friendly}' alerts? "
            "This can only be undone from settings."
        ),
        "action_hint_high": "Urgent: check and secure it now.",
        "action_hint_medium": "Check soon and secure it if unexpected.",
        "action_hint_low": "Review when convenient.",
    },
    "cs": {
        "severity_title_high": "Bezpečnostní výstraha",
        "severity_title_medium": "Upozornění z domova",
        "severity_title_low": "Novinka z domova",
        "type_open_entry_while_away": "Otevřený vstup v nepřítomnosti",
        "type_open_entry_at_night": "Otevřený vstup v noci",
        "type_open_any_window_at_night_while_away": "Otevřené okno v noci",
        "type_motion_detected_at_night_while_away": ("Pohyb v noci v nepřítomnosti"),
        "type_motion_detected_while_away": "Pohyb v nepřítomnosti",
        "type_unlocked_lock_at_night": "Zámek ponechán odemčený",
        "type_camera_entry_unsecured": "Pohyb u nezajištěného vstupu",
        "type_alarm_disarmed_during_external_threat": (
            "Pohyb venku při vypnutém alarmu"
        ),
        "type_appliance_power_duration": "Doba provozu spotřebiče",
        "subtitle_appliance_finished": "{appliance} dokončeno",
        "subtitle_appliance_cycle_complete": "Cyklus spotřebiče dokončen",
        "subtitle_entry_open_alarm_disarmed": "{entry_name} otevřeno, alarm vypnutý",
        "subtitle_power_deviation": "{appliance}: odběr {direction}, než se čekalo",
        "subtitle_reading_deviation": (
            "{appliance}: hodnota {direction}, než se čekalo"
        ),
        "direction_lower": "nižší",
        "direction_higher": "vyšší",
        "fallback_entry": "Vstup",
        "fallback_sensor": "Senzor",
        "fallback_unknown_entity": "Neznámá entita",
        "fallback_message": "{summary}: {entity}. {action_hint}",
        "persistent_fallback": (
            "{summary} (závažnost {severity}) pro {entities}. {hint}"
        ),
        "batch_title": "Novinka z domova",
        "batch_message": "{count} novinek z domova: {type_summary}.",
        "digest_title": "Denní přehled Sentinelu",
        "digest_message": (
            "Sentinel: {count} upozornění za posledních 24 h ({sev_summary})."
        ),
        "severity_word_high": "vysoká",
        "severity_word_medium": "střední",
        "severity_word_low": "nízká",
        "snooze_confirm_title": "Potvrdit trvalé ztlumení",
        "snooze_confirm_message": (
            "Trvale potlačit upozornění typu '{friendly}'? "
            "Lze vrátit zpět pouze v nastavení."
        ),
        "action_hint_high": "Naléhavé: zkontrolujte a zajistěte to hned.",
        "action_hint_medium": "Zkontrolujte brzy a zajistěte, pokud je to neobvyklé.",
        "action_hint_low": "Projděte si to, až budete mít čas.",
    },
}

_DEFAULT_LANGUAGE = "en"


def _resolve_language(hass: HomeAssistant | None) -> str:
    """Resolve the message language from the HA instance's configured language."""
    if hass is None:
        return _DEFAULT_LANGUAGE
    # Guard the config attribute itself, not just language: notification
    # dispatch must never crash on language resolution, and lightweight
    # test doubles may not carry a config object at all.
    config = getattr(hass, "config", None)
    language = getattr(config, "language", None) or _DEFAULT_LANGUAGE
    base = language.split("-", 1)[0].lower()
    return base if base in _MESSAGES else _DEFAULT_LANGUAGE


def notif_msg(hass: HomeAssistant | None, key: str, /, **kwargs: Any) -> str:
    """
    Return the localized Sentinel notification-chrome string for ``key``.

    Falls back to the English string if the resolved language is
    missing the key, and to the key itself if English is missing it
    too (so a bug here shows up as an odd sentence, not a crash).
    That promise also covers formatting: this runs on the last hop of
    the notification pipeline, so a template whose placeholders drift
    from what the call site supplies degrades to the English (then
    unformatted) string instead of raising and losing the alert.
    """
    language = _resolve_language(hass)
    table = _MESSAGES.get(language, _MESSAGES[_DEFAULT_LANGUAGE])
    english = _MESSAGES[_DEFAULT_LANGUAGE].get(key)
    template = table.get(key) or english
    if template is None:
        # Unknown message id: return it verbatim and never treat it as a
        # format template — the id may be data-derived (e.g. a stored
        # severity word in the daily digest).
        return key
    if not kwargs:
        return template
    for candidate in (template, english):
        if candidate is None:
            continue
        try:
            return candidate.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            continue
    return template

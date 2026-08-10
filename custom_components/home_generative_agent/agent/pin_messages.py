"""
Localized user-facing strings for the critical-action PIN flow.

The critical-action PIN gate (added in v3.26.0) returns fixed strings
directly as tool results / ToolMessage content. These strings are shown
to the user in the conversation and are *not* covered by Home
Assistant's ``strings.json`` / ``translations/*.json`` mechanism, since
they never pass through a config flow or entity translation key - they
are produced at runtime by plain Python functions in ``tools.py`` and
``graph.py``.

This module gives that small, fixed set of strings the same "pick a
language, fall back to English" treatment already used elsewhere in
this integration (see ``CONF_SENTINEL_RESPONSE_LANGUAGE`` and
``CONF_VLM_RESPONSE_LANGUAGE``). Unlike those two options, these
messages are deterministic (not LLM-authored), so a static lookup
table is used instead of an LLM call - there is nothing to translate
on the fly and no reason to pay for or cache a completion.

Add a new language by adding a key to ``_MESSAGES`` with the same set
of message ids as ``"en"``. Missing keys in a non-English language
fall back to the English string automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# Message ids are stable identifiers, not the English text itself, so
# wording can be tuned per-language independently.
_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "automation_pin_unset": (
            "This automation performs a critical action, but the "
            "critical-action PIN is enabled without a PIN being set, so "
            "it cannot be confirmed. Set a PIN in the integration "
            "options and try again."
        ),
        "automation_pending_store_unavailable": (
            "Unable to confirm this automation right now; please try again."
        ),
        "automation_requires_pin_reason": (
            "This automation requires PIN confirmation because {reason}: {calls}."
        ),
        "direct_requires_pin_reason": "Critical action requires PIN confirmation.",
        "pending_action_not_found_or_expired": "Pending action not found or expired.",
        "pending_action_not_found": "Pending action not found.",
        "pending_action_wrong_user": (
            "Pending action belongs to a different user; please re-run the request."
        ),
        "pending_action_invalid": "Pending action is invalid; please try again.",
        "pending_action_expired": "Pending action expired; please re-run the request.",
        "pin_not_configured": "No PIN configured; cannot confirm the action.",
        "pin_invalid_format": "Invalid PIN. Use {min_len}-{max_len} digits.",
        "pin_too_many_attempts": (
            "Too many incorrect attempts; please re-run the request."
        ),
        "pin_incorrect": "Incorrect PIN. Action not executed.",
        "confirmation_unable_to_process": "Unable to process the confirmation.",
        "pending_action_missing_tool_name": (
            "Pending action is invalid; missing tool name."
        ),
        "ha_llm_api_unavailable": "Home Assistant LLM API unavailable.",
        "action_execute_failed": "Failed to execute action: {err}",
        "pending_automation_invalid": (
            "Pending automation is invalid; please re-run the request."
        ),
        "automation_install_failed": (
            "Failed to install the automation: {err}. "
            "The confirmation was used up; please request the automation again."
        ),
    },
    "cs": {
        "automation_pin_unset": (
            "Tato automatizace provádí kritickou akci, ale PIN pro "
            "kritické akce je povolen bez nastaveného PIN kódu, takže ji "
            "nelze potvrdit. Nastavte PIN v možnostech integrace a "
            "zkuste to znovu."
        ),
        "automation_pending_store_unavailable": (
            "Tuto automatizaci nyní nelze potvrdit; zkuste to prosím znovu."
        ),
        "automation_requires_pin_reason": (
            "Tato automatizace vyžaduje potvrzení PIN kódem, protože {reason}: {calls}."
        ),
        "direct_requires_pin_reason": "Kritická akce vyžaduje potvrzení PIN kódem.",
        "pending_action_not_found_or_expired": (
            "Čekající akce nebyla nalezena nebo vypršela její platnost."
        ),
        "pending_action_not_found": "Čekající akce nebyla nalezena.",
        "pending_action_wrong_user": (
            "Čekající akce patří jinému uživateli; zopakujte prosím požadavek."
        ),
        "pending_action_invalid": "Čekající akce je neplatná; zkuste to prosím znovu.",
        "pending_action_expired": (
            "Platnost čekající akce vypršela; zopakujte prosím požadavek."
        ),
        "pin_not_configured": "Není nastaven žádný PIN; akci nelze potvrdit.",
        "pin_invalid_format": "Neplatný PIN. Použijte {min_len}-{max_len} číslic.",
        "pin_too_many_attempts": (
            "Příliš mnoho neúspěšných pokusů; zopakujte prosím požadavek."
        ),
        "pin_incorrect": "Nesprávný PIN. Akce nebyla provedena.",
        "confirmation_unable_to_process": "Potvrzení se nepodařilo zpracovat.",
        "pending_action_missing_tool_name": (
            "Čekající akce je neplatná; chybí název nástroje."
        ),
        "ha_llm_api_unavailable": "LLM API Home Assistantu není k dispozici.",
        "action_execute_failed": "Akci se nepodařilo provést: {err}",
        "pending_automation_invalid": (
            "Čekající automatizace je neplatná; zopakujte prosím požadavek."
        ),
        "automation_install_failed": (
            "Automatizaci se nepodařilo nainstalovat: {err}. "
            "Potvrzení bylo použito; požádejte prosím o automatizaci znovu."
        ),
    },
}

_DEFAULT_LANGUAGE = "en"


def _resolve_language(hass: HomeAssistant | None) -> str:
    """Resolve the message language from the HA instance's configured language."""
    if hass is None:
        return _DEFAULT_LANGUAGE
    language = getattr(hass.config, "language", None) or _DEFAULT_LANGUAGE
    # HA language codes may include a region suffix (e.g. "cs-CZ"); the
    # message table is keyed by the base language only.
    base = language.split("-", 1)[0].lower()
    return base if base in _MESSAGES else _DEFAULT_LANGUAGE


def pin_msg(hass: HomeAssistant | None, key: str, /, **kwargs: Any) -> str:
    """
    Return the localized PIN-flow message for ``key``.

    Falls back to the English string if the resolved language is
    missing the key, and to the key itself (so a bug here shows up as
    an odd sentence rather than a crash) if English is missing it too.
    """
    language = _resolve_language(hass)
    table = _MESSAGES.get(language, _MESSAGES[_DEFAULT_LANGUAGE])
    template = table.get(key) or _MESSAGES[_DEFAULT_LANGUAGE].get(key) or key
    return template.format(**kwargs) if kwargs else template

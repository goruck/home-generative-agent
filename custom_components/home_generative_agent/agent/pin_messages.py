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
        "pin_awaiting_user_reply": (
            "Not confirmed: the PIN must be in the user's current message."
        ),
        "pin_retry_next_message": (
            "The user can say the PIN again in their next message."
        ),
        "pin_say_digits": (
            "Ask the user to say the PIN digits one by one in their next message."
        ),
        "pin_request_again": (
            "That was the last reply this PIN prompt takes; the user has to make "
            "the request again."
        ),
        "pin_reply_used": (
            "Not confirmed: the PIN in the user's current message was already used. "
            "Tell the user what happened; they can ask again."
        ),
        "action_execute_failed": (
            "Failed to execute action: {err}. "
            "The confirmation was used up; tell the user the action failed "
            "and that they can ask again."
        ),
        "pending_automation_invalid": (
            "Pending automation is invalid; please re-run the request."
        ),
        "action_outcome_unknown": (
            "The action may have run, but it ended with an unexpected error: "
            "{err}. The confirmation was used up. Do not repeat the action; "
            "tell the user to check the device before asking again."
        ),
        "automation_outcome_unknown": (
            "The automation may have been installed, but the install ended with "
            "an unexpected error: {err}. The confirmation was used up. Do not "
            "repeat it; tell the user to check their automations first."
        ),
        "pin_reply_to_prompt_only": (
            "Not confirmed: the PIN is accepted only in the user's reply to "
            "this action's PIN prompt. Ask the user for the PIN now."
        ),
        "pin_one_action_at_a_time": (
            "Not requested: another protected action is already waiting for the "
            "user's PIN. Ask for that PIN first; the user can request this one "
            "afterwards."
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
        "pin_awaiting_user_reply": (
            "Nepotvrzeno: PIN musí být v aktuální zprávě uživatele."
        ),
        "pin_retry_next_message": "Uživatel může PIN zopakovat v další zprávě.",
        "pin_say_digits": (
            "Požádejte uživatele, aby v další zprávě řekl číslice PINu jednu po druhé."
        ),
        "pin_request_again": (
            "To byla poslední odpověď, kterou tato výzva k PINu přijímá; "
            "uživatel musí požadavek zopakovat."
        ),
        "pin_reply_used": (
            "Nepotvrzeno: PIN z aktuální zprávy uživatele už byl použit. "
            "Sdělte uživateli, co se stalo; může požádat znovu."
        ),
        "action_execute_failed": (
            "Akci se nepodařilo provést: {err}. "
            "Potvrzení bylo použito; sdělte uživateli, že akce selhala "
            "a že může požádat znovu."
        ),
        "pending_automation_invalid": (
            "Čekající automatizace je neplatná; zopakujte prosím požadavek."
        ),
        "action_outcome_unknown": (
            "Akce možná proběhla, ale skončila neočekávanou chybou: {err}. "
            "Potvrzení bylo použito. Akci neopakujte; požádejte uživatele, "
            "aby před dalším požadavkem zkontroloval zařízení."
        ),
        "automation_outcome_unknown": (
            "Automatizace možná byla nainstalována, ale instalace skončila "
            "neočekávanou chybou: {err}. Potvrzení bylo použito. Neopakujte ji; "
            "požádejte uživatele, aby nejdřív zkontroloval své automatizace."
        ),
        "pin_reply_to_prompt_only": (
            "Nepotvrzeno: PIN se přijímá jen v odpovědi uživatele na výzvu "
            "k PINu pro tuto akci. Požádejte uživatele o PIN."
        ),
        "pin_one_action_at_a_time": (
            "Nevyžádáno: na PIN uživatele už čeká jiná chráněná akce. Nejdřív "
            "požádejte o PIN k té čekající akci; o tuto akci může uživatel "
            "požádat potom."
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

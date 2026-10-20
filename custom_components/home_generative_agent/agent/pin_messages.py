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

The module also builds the one deterministic sentence the user is told to
trust: ``describe_pending_action`` names a held action from the call itself
(the verb HA's handler runs for the domain, the target or the scope HA will
act on), and ``conversation.py`` joins it to every reply as the
``pin_confirm_line`` / ``pin_done_line`` family. It imports from
``.helpers``, so ``helpers`` must never import this module.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .automation_pin import DATA_CONTAINER_KEYS
from .helpers import (
    INTENT_NAME_ALL,
    SET_POSITION_INTENT,
    STOP_MOVING_INTENT,
    TOOL_NAMESPACE_SEP,
    base_tool_name,
    intent_service_calls,
    is_actuating_intent,
    is_device_intent,
    is_on_off_intent,
    minimal_payload_for_domain,
    normalize_intent_for_lock,
    sanitize_tool_text,
)
from .indirect_pin import (
    bare_script_tool_entity_id,
    intent_script_handlers,
    script_tool_entity_id,
)
from .rule_entities import relay_wrappers_reached

LOGGER = logging.getLogger(__name__)

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
        "pin_reply_next_message": (
            "Not confirmed: the PIN is accepted only in the user's reply to "
            "this action's PIN prompt. The user is being asked for the PIN now; "
            "their next message is the reply. Confirm with the PIN from it."
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
        "pin_confirm_line": "Your PIN will confirm: {action}.",
        "pin_incorrect_line": "That PIN was incorrect.",
        "pin_request_again_line": "Make the request again.",
        "pin_done_line": "PIN accepted: {action}.",
        "pin_failed_line": "PIN accepted, but {action} failed.",
        "pin_unknown_line": "PIN accepted; {action} may have run. Check the device.",
        # Ends a reply that is only the fixed line, so a satellite keeps
        # listening for the answer.
        "pin_ask_line": "What is your PIN?",
        "action_lock": "lock {target}",
        "action_unlock": "unlock {target}",
        "action_open": "open {target}",
        "action_relay_controls": "controls {target}",
        "action_close": "close {target}",
        "action_turn_on": "turn on {target}",
        "action_turn_off": "turn off {target}",
        "action_set_position": "set {target} to {position}%",
        "action_press": "press {target}",
        "action_intent_script": "run the voice command {target}",
        "action_starts_automation": "starts the automation {automations}",
        "action_stop": "stop {target}",
        "action_service": "{service} {target}",
        "action_automation": "add the automation {target}",
        "action_unnamed_automation": "add an unnamed automation",
        "action_automation_runs": "{action}, which runs {calls}",
        "action_calls_more": "{calls} and {count} more",
        "action_unnamed_target": "an unnamed device",
        # The target as HA reads it: one named device, or the scope of a
        # call that names none (every device of the domain, in an area, on
        # a floor).
        "target_named": '"{name}"',
        # A device named by its entity id: its own name, then the id HA runs.
        "target_entity": '"{name}" ({entity_id})',
        "target_every": "{kind}",
        "target_every_in_area": "{kind} in {area}",
        "target_every_on_floor": "{kind} on floor {floor}",
        "kind_lock": "every lock",
        "kind_cover": "every cover",
        "kind_valve": "every valve",
        "kind_domain": "every {domain} device",
        "kind_device": "every device",
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
        "pin_reply_next_message": (
            "Nepotvrzeno: PIN se přijímá jen v odpovědi uživatele na výzvu "
            "k PINu pro tuto akci. Uživatel je o PIN právě žádán; jeho další "
            "zpráva je odpověď. Potvrďte PINem z ní."
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
        "pin_confirm_line": "Váš PIN potvrdí: {action}.",
        "pin_incorrect_line": "Ten PIN byl nesprávný.",
        "pin_request_again_line": "Zopakujte požadavek.",
        "pin_done_line": "PIN přijat: {action}.",
        "pin_failed_line": "PIN přijat, ale {action} se nepodařilo.",
        "pin_unknown_line": (
            "PIN přijat; {action} možná proběhlo. Zkontrolujte zařízení."
        ),
        "pin_ask_line": "Jaký je váš PIN?",
        "action_lock": "zamknout {target}",
        "action_unlock": "odemknout {target}",
        "action_open": "otevřít {target}",
        "action_relay_controls": "ovládá {target}",
        "action_close": "zavřít {target}",
        "action_turn_on": "zapnout {target}",
        "action_turn_off": "vypnout {target}",
        "action_set_position": "nastavit {target} na {position} %",
        "action_press": "stisknout {target}",
        "action_intent_script": "spustit hlasový příkaz {target}",
        "action_starts_automation": "spustí automatizaci {automations}",
        "action_stop": "zastavit {target}",
        "action_service": "{service} {target}",
        "action_automation": "přidat automatizaci {target}",
        "action_unnamed_automation": "přidat nepojmenovanou automatizaci",
        "action_automation_runs": "{action}, která spouští {calls}",
        "action_calls_more": "{calls} a dalších {count}",
        "action_unnamed_target": "nepojmenované zařízení",
        "target_named": "„{name}“",
        "target_entity": "„{name}“ ({entity_id})",
        "target_every": "{kind}",
        "target_every_in_area": "{kind} v oblasti {area}",
        "target_every_on_floor": "{kind} v patře {floor}",
        "kind_lock": "všechny zámky",
        "kind_cover": "všechny rolety a vrata",
        "kind_valve": "všechny ventily",
        "kind_domain": "všechna zařízení typu {domain}",
        "kind_device": "všechna zařízení",
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


# Longest target name the action line repeats. The name comes from the model.
_ACTION_TARGET_MAX_LEN = 80
# Longest service name the line repeats, for a call that is not an intent.
_ACTION_SERVICE_MAX_LEN = 40
# Critical calls named for an automation, and the length of each. More calls
# are counted ("and 2 more"), and a longer call ends in an ellipsis, so an
# omission is always visible in the line.
_AUTOMATION_CALLS_MAX = 5
_AUTOMATION_CALL_MAX_LEN = 60
_ELLIPSIS = "\u2026"

# How Home Assistant's OnOffIntentHandler runs HassTurnOn / HassTurnOff by
# domain: (on, off). Any other domain is turned on or off. HA refuses to turn
# a button off, so that call never runs and needs no line of its own.
_ON_OFF_VERBS: dict[str, tuple[str, str]] = {
    "lock": ("action_lock", "action_unlock"),
    "cover": ("action_open", "action_close"),
    "valve": ("action_open", "action_close"),
    "button": ("action_press", "action_turn_off"),
    "input_button": ("action_press", "action_turn_off"),
}
# The order the actions of a call over several kinds of device are named
# in: the ones a PIN rule is most likely about first.
_VERB_PRIORITY = ("lock", "cover", "valve", "button", "input_button")
_DEFAULT_VERBS = ("action_turn_on", "action_turn_off")
# The kind of an action over several kinds of device, named "every device".
_SEVERAL_KINDS = "*"
_KIND_KEYS = {"lock": "kind_lock", "cover": "kind_cover", "valve": "kind_valve"}


def _one_line(value: Any, limit: int, *, mark_cut: bool = False) -> str:
    """Return model-chosen text as one bounded, printable line."""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v)
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if mark_cut and len(text) > limit:
        return sanitize_tool_text(text[: limit - 1], limit=limit) + _ELLIPSIS
    return sanitize_tool_text(text, limit=limit)


def _first(value: Any) -> str:
    """Return the first entry of a list value, or the value, as text."""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "")


def _named(hass: HomeAssistant | None, name: str) -> str:
    return pin_msg(hass, "target_named", name=name)


def _declared_domain(tool_name: str, tool_args: Mapping[str, Any]) -> str:
    """
    Return the domain an intent declares, for a line built without a match.

    A declared `lock` domain (the lock normalizer), else the domain argument,
    else the domain of an entity id the model gave. Never from the name: this
    line is built only when HA's matcher found nothing, so there is no device
    the name could be read from (#782).
    """
    normalized = normalize_intent_for_lock(tool_name, dict(tool_args), None)
    domain = _first(normalized.get("domain")).lower()
    if domain:
        return domain
    entity_id = _first(tool_args.get("entity_id") or tool_args.get("entity_ids"))
    return entity_id.split(".", 1)[0].lower() if "." in entity_id else ""


def _intent_actions(
    hass: HomeAssistant | None, tool_name: str, tool_args: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """
    Return one (verb, kind) domain pair per distinct action an intent takes.

    From the entities HA's matcher finds, the same resolution the gate uses,
    so a lock named only "Front Door" reads "unlock" (#770) and an entity id
    the model added, which HA ignores, cannot change the verb. Kinds that
    share a verb are one action; one that does something else is its own,
    so turning on a garage with a lock and the garage door says both "lock"
    and "open", never one for the other. The kind names the target only
    when the action covers one kind ("every device" otherwise). With nothing
    matched, the domain the call declares.
    """
    resolved = intent_service_calls(hass, tool_name, tool_args) if hass else None
    domains = {call.domain for call in resolved or []}
    if not domains:
        domain = _declared_domain(tool_name, tool_args)
        return [(domain, domain)]
    rank = {domain: index for index, domain in enumerate(_VERB_PRIORITY)}
    by_verb: dict[tuple[str, str], list[str]] = {}
    for domain in sorted(domains, key=lambda d: (rank.get(d, len(rank)), d)):
        by_verb.setdefault(_ON_OFF_VERBS.get(domain, _DEFAULT_VERBS), []).append(domain)
    return [
        (group[0], group[0] if len(group) == 1 else _SEVERAL_KINDS)
        for group in by_verb.values()
    ]


def _single_kind(
    hass: HomeAssistant | None, tool_name: str, tool_args: Mapping[str, Any]
) -> str:
    """Return the one kind an intent with a single verb acts on, or ""."""
    actions = _intent_actions(hass, tool_name, tool_args)
    kinds = {kind for _, kind in actions}
    return kinds.pop() if len(actions) == 1 and len(kinds) == 1 else _SEVERAL_KINDS


def _slot_line(value: Any) -> str:
    """Return a slot value as one line, read the way HA's slot schema reads it."""
    if value is None or isinstance(value, (list, dict)):
        return ""
    return _one_line(str(value), _ACTION_TARGET_MAX_LEN)


# Quote marks a device's own name may carry, read in the line as its end.
_QUOTE_MARKS = str.maketrans(dict.fromkeys('"“”„', "'"))


def _entity_friendly_name(hass: HomeAssistant | None, raw_name: Any) -> str:
    """
    Return the name of the device a name slot gives by its entity id, or "".

    HA reads a name as an entity id only when it is the id exactly, so the
    slot is looked up as sent, before it is shortened for the line. Quote
    marks in the device's name are shown as apostrophes, so a name cannot
    close the quote and read as more of the line.
    """
    states = getattr(hass, "states", None)
    if states is None or not isinstance(raw_name, str):
        return ""
    state = states.get(raw_name)
    if state is None or state.entity_id != raw_name:
        return ""
    return _slot_line(state.name.translate(_QUOTE_MARKS))


def _intent_target(
    hass: HomeAssistant | None, tool_args: Mapping[str, Any], domain: str
) -> str:
    """
    Name what Home Assistant will act on for an actuating intent.

    Read from the payload the call is sent as (`minimal_payload_for_domain`,
    which the normal path and the PIN confirmation both apply), so the line
    and the call cannot drift: HA's intents read the name, area and floor
    slots, and the payload sends an entity id the model gave as the name.
    With no name, every device of the kind acts (every device, when the
    matches are of several kinds), in the area or on the floor if one is
    given, and the line says so rather than naming a device it will not
    single out.
    """
    payload = minimal_payload_for_domain(dict(tool_args))
    raw_name = payload.get("name")
    name = "" if raw_name == INTENT_NAME_ALL else _slot_line(raw_name)
    if name:
        friendly = _entity_friendly_name(hass, raw_name)
        if friendly and friendly != name:
            return pin_msg(hass, "target_entity", name=friendly, entity_id=name)
        return _named(hass, name)
    area = _slot_line(payload.get("area"))
    floor = _slot_line(payload.get("floor"))
    if not domain and not area and not floor:
        return pin_msg(hass, "action_unnamed_target")
    if domain and domain != _SEVERAL_KINDS:
        kind_key = _KIND_KEYS.get(domain, "kind_domain")
        kind = pin_msg(hass, kind_key, domain=sanitize_tool_text(domain, limit=40))
    else:
        kind = pin_msg(hass, "kind_device")
    if area:
        return pin_msg(hass, "target_every_in_area", kind=kind, area=area)
    if floor:
        return pin_msg(hass, "target_every_on_floor", kind=kind, floor=floor)
    return pin_msg(hass, "target_every", kind=kind)


def _position_text(value: Any) -> str:
    """Return a held position as HA will run it (`Coerce(int)`), 0 included."""
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return _one_line(value, _ACTION_SERVICE_MAX_LEN)


def _describe_automation(hass: HomeAssistant | None, action: Mapping[str, Any]) -> str:
    """Name an automation by its alias and the critical calls it would run."""
    config = action.get("automation_config")
    alias = _one_line(
        config.get("alias") if isinstance(config, Mapping) else None,
        _ACTION_TARGET_MAX_LEN,
    )
    text = (
        pin_msg(hass, "action_automation", target=_named(hass, alias))
        if alias
        else pin_msg(hass, "action_unnamed_automation")
    )
    raw_calls = action.get("critical_calls")
    calls = [
        _one_line(call, _AUTOMATION_CALL_MAX_LEN, mark_cut=True)
        for call in (raw_calls if isinstance(raw_calls, list) else [])
    ]
    calls = [call for call in calls if call]
    if not calls:
        return text
    shown = ", ".join(calls[:_AUTOMATION_CALLS_MAX])
    if len(calls) > _AUTOMATION_CALLS_MAX:
        shown = pin_msg(
            hass,
            "action_calls_more",
            calls=shown,
            count=len(calls) - _AUTOMATION_CALLS_MAX,
        )
    return pin_msg(hass, "action_automation_runs", action=text, calls=shown)


def _intent_script_line(
    hass: HomeAssistant | None, tool_name: str, tool_args: Mapping[str, Any]
) -> str | None:
    """
    Name the intent script a tool runs, with everything the model passed (#779).

    Only for a tool an intent script owns: its own ``intent_script__`` tool,
    or one named by it (an intent script taking over HassTurnOn) that is not
    a script's tool and names no service of its own. Every intent script
    that may be behind it is named. Every argument is shown, since Home
    Assistant hands each to the intent script, and one of them may pick the
    device.
    """
    handlers = [h for h in intent_script_handlers(hass, tool_name) if h]
    if not handlers or script_tool_entity_id(tool_name) is not None:
        return None
    # A tool another API offers under the same name is a service tool when
    # it names a service. `domain` is an intent slot, and the lock
    # normalizer adds `service` to an on/off intent, so neither makes an
    # intent's own tool a service call.
    owned = (
        TOOL_NAMESPACE_SEP + "intent_script" + TOOL_NAMESPACE_SEP
        in TOOL_NAMESPACE_SEP + tool_name
        or is_actuating_intent(tool_name)
        or is_device_intent(tool_name)
    )
    if not owned and tool_args.get("service"):
        return None
    names = sorted(
        {str(getattr(h, "intent_type", "") or "") for h in handlers} - {""}
    ) or [base_tool_name(tool_name)]
    line = pin_msg(
        hass,
        "action_intent_script",
        target=" / ".join(
            _named(hass, _one_line(name, _ACTION_TARGET_MAX_LEN)) for name in names
        ),
    )
    slots = [
        f"{key}: {_one_line(str(value), _ACTION_TARGET_MAX_LEN)}"
        for key, value in sorted(tool_args.items())
        if value not in (None, "", [], {})
    ]
    return f"{line} ({', '.join(slots)})" if slots else line


def describe_pending_action(
    hass: HomeAssistant | None, action: Mapping[str, Any]
) -> str:
    """
    Name the action a PIN confirms, and what a relay it switches controls.

    A call on the switch under a switch_as_x garage cover or lock (#794) reads
    as a switch ("turn off Garage Relay"); the line adds what it moves
    ("(controls Garage Door)"), so the user knows what the PIN allows.
    """
    line = _describe_line(hass, action)
    starts = [str(name) for name in action.get("starts") or [] if name]
    if starts:
        # The call is held for an automation it would start (#814): say so,
        # or "turn on Let In" reads as harmless.
        line = (
            f"{line} ("
            + pin_msg(
                hass,
                "action_starts_automation",
                automations=", ".join(
                    f'"{_one_line(name, _ACTION_TARGET_MAX_LEN)}"' for name in starts
                ),
            )
            + ")"
        )
    if hass is None or str(action.get("tool_name") or "") == "add_automation":
        return line
    try:
        effects = _relay_effects(hass, action)
    except Exception:
        # The note only adds to the line: it must never cost the prompt, nor
        # turn a confirmed action that already ran into an error.
        LOGGER.exception("Critical Action PIN: could not name what a relay moves")
        effects = []
    return f"{line} ({', '.join(effects)})" if effects else line


def _named_entity_ids(tool_args: Mapping[str, Any]) -> set[str]:
    """Return the entity ids a call names, at the top level or in a container."""
    found: set[str] = set()
    sources = [tool_args] + [
        tool_args[key]
        for key in ("target", *DATA_CONTAINER_KEYS)
        if isinstance(tool_args.get(key), Mapping)
    ]
    for source in sources:
        for key in ("entity_id", "entity_ids"):
            value = source.get(key)
            values = value if isinstance(value, list) else [value]
            found |= {
                part.strip().lower()
                for item in values
                if isinstance(item, str)
                for part in item.split(",")
                if part.strip()
            }
    return found


def _relay_effects(hass: HomeAssistant, action: Mapping[str, Any]) -> list[str]:
    """Return 'controls <cover>' for each wrapper the call moves through a relay."""
    tool_name = str(action.get("tool_name") or "")
    raw_args = action.get("tool_args")
    tool_args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    targets = _named_entity_ids(tool_args)
    if is_on_off_intent(tool_name):
        calls = intent_service_calls(hass, tool_name, tool_args) or []
        targets |= {call.entity_id.lower() for call in calls if call.entity_id}
    effects: list[str] = []
    for wrapper in relay_wrappers_reached(hass, targets):
        state = hass.states.get(wrapper)
        name = state.name if state is not None else wrapper
        # Not "open": a relay moves a door either way, and the call may be the
        # one that closes it.
        effects.append(
            pin_msg(
                hass,
                "action_relay_controls",
                target=_named(hass, _one_line(name, _ACTION_TARGET_MAX_LEN)),
            )
        )
    return effects


def _describe_line(  # noqa: PLR0911
    hass: HomeAssistant | None, action: Mapping[str, Any]
) -> str:
    """
    Name the action a PIN confirms, from what will run, not from the model.

    An on/off intent is named the way Home Assistant runs it for the domain:
    HassTurnOn locks, opens or turns on; HassTurnOff unlocks, closes or turns
    off, whatever service the model added. Field test 2026-10-15: the model
    asked for the PIN "to lock the door" while the held call was HassTurnOff,
    and the PIN unlocked it. The target is the one HA will act on, or the
    scope of a call that names none.
    """
    tool_name = str(action.get("tool_name") or "")
    if tool_name == "add_automation":
        return _describe_automation(hass, action)

    raw_args = action.get("tool_args")
    tool_args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    if (line := _intent_script_line(hass, tool_name, tool_args)) is not None:
        return line
    if (script_id := script_tool_entity_id(tool_name)) is not None:
        return _script_line(hass, script_id)
    # A tool named like a script (HA before 2026.9 named script tools so) is
    # held for the script or for the call its arguments make, so the line
    # names the script, and the call too when it makes one.
    bare_script = bare_script_tool_entity_id(tool_name, hass)
    if bare_script is not None and not (
        tool_args.get("domain") or tool_args.get("service")
    ):
        return _script_line(hass, bare_script)

    if base_tool_name(tool_name) == SET_POSITION_INTENT:
        return pin_msg(
            hass,
            "action_set_position",
            target=_intent_target(
                hass, tool_args, _single_kind(hass, tool_name, tool_args)
            ),
            position=_position_text(tool_args.get("position")),
        )
    if base_tool_name(tool_name) == STOP_MOVING_INTENT:
        return pin_msg(
            hass,
            "action_stop",
            target=_intent_target(
                hass, tool_args, _single_kind(hass, tool_name, tool_args)
            ),
        )
    if is_on_off_intent(tool_name):
        turns_on = base_tool_name(tool_name) == "HassTurnOn"
        parts = []
        for verb, kind in _intent_actions(hass, tool_name, tool_args):
            on_key, off_key = _ON_OFF_VERBS.get(verb, _DEFAULT_VERBS)
            parts.append(
                pin_msg(
                    hass,
                    on_key if turns_on else off_key,
                    target=_intent_target(hass, tool_args, kind),
                )
            )
        return ", ".join(parts)

    target = _one_line(
        tool_args.get("name")
        or tool_args.get("entity_id")
        or tool_args.get("entity_ids")
        or tool_args.get("area"),
        _ACTION_TARGET_MAX_LEN,
    )
    service = str(tool_args.get("service") or base_tool_name(tool_name))
    line = pin_msg(
        hass,
        "action_service",
        service=sanitize_tool_text(
            service.replace("_", " "), limit=_ACTION_SERVICE_MAX_LEN
        ),
        target=_named(hass, target)
        if target
        else pin_msg(hass, "action_unnamed_target"),
    )
    if bare_script is not None:
        return ", ".join([_script_line(hass, bare_script), line])
    return line


def _script_line(hass: HomeAssistant | None, script_id: str) -> str:
    """
    Name running a script, from Home Assistant.

    A script's arguments are free-form fields the model fills in, so the line
    names the script from Home Assistant, never from them.
    """
    state = hass.states.get(script_id) if hass is not None else None
    name = state.name if state is not None else script_id
    return pin_msg(
        hass,
        "action_turn_on",
        target=_named(hass, _one_line(name, _ACTION_TARGET_MAX_LEN)),
    )

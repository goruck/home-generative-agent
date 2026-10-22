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
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import intent as ha_intent
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers.service import async_get_cached_service_description

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
    call_service_names,
    intent_script_handlers,
    named_services,
    script_tool_entity_id,
)
from .rule_entities import relay_wrappers_reached
from .template_actions import INLINE_STATE_KEYS

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
        # A state a scene call sets (#786), or one whose value is not shown.
        "action_set_state": "set {target} to {state}",
        "action_change": "change {target}",
        "action_create_scene": "create the scene {scene} that sets {states}",
        "target_to_state": "{target} to {state}",
        # The devices a call names, when it also reaches others it does not.
        "target_and_others": "{targets} and other devices",
        "target_unnamed_devices": "devices it does not name",
        # What the agent passed to a script's device fields (#787).
        "action_script_targets": "{action} ({targets})",
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
        # A named device the call limits to an area or floor (#784): two
        # locks named "Door" read differently.
        "target_in_area": "{target} in {area}",
        "target_on_floor": "{target} on floor {floor}",
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
        "action_set_state": "nastavit {target} na {state}",
        "action_change": "změnit {target}",
        "action_create_scene": "vytvořit scénu {scene}, která nastaví {states}",
        "target_to_state": "{target} na {state}",
        "target_and_others": "{targets} a další zařízení",
        "target_unnamed_devices": "zařízení, která nejmenuje",
        "action_script_targets": "{action} ({targets})",
        "action_unnamed_target": "nepojmenované zařízení",
        "target_named": "„{name}“",
        "target_entity": "„{name}“ ({entity_id})",
        "target_every": "{kind}",
        "target_every_in_area": "{kind} v oblasti {area}",
        "target_every_on_floor": "{kind} v patře {floor}",
        "target_in_area": "{target} v oblasti {area}",
        "target_on_floor": "{target} v patře {floor}",
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
_SCENE_DOMAIN = "scene"
_SCENE_INLINE_SERVICES = frozenset({"apply", "create"})
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
    return _slot_line(str(state.name).translate(_QUOTE_MARKS))


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
    area = _place(hass, payload.get("area"), "area")
    floor = _place(hass, payload.get("floor"), "floor")
    if name:
        friendly = _entity_friendly_name(hass, raw_name)
        if friendly and friendly != name:
            target = pin_msg(hass, "target_entity", name=friendly, entity_id=name)
        else:
            target = _named(hass, name)
        # HA matches the name within the area and floor the call gives, so
        # the line names them too (#784), as HA resolves them: by id, name or
        # alias, every match.
        if area:
            target = pin_msg(hass, "target_in_area", target=target, area=area)
        if floor:
            target = pin_msg(hass, "target_on_floor", target=target, floor=floor)
        return target
    if not domain and not area and not floor:
        return pin_msg(hass, "action_unnamed_target")
    if domain and domain != _SEVERAL_KINDS:
        kind_key = _KIND_KEYS.get(domain, "kind_domain")
        kind = pin_msg(hass, kind_key, domain=sanitize_tool_text(domain, limit=40))
    else:
        kind = pin_msg(hass, "kind_device")
    if area:
        target = pin_msg(hass, "target_every_in_area", kind=kind, area=area)
        if floor:
            target = pin_msg(hass, "target_on_floor", target=target, floor=floor)
        return target
    if floor:
        return pin_msg(hass, "target_every_on_floor", kind=kind, floor=floor)
    return pin_msg(hass, "target_every", kind=kind)


def _place(hass: HomeAssistant | None, value: Any, kind: str) -> str:
    """
    Name an area or floor slot as HA matches it, with or without a name.

    HA resolves the slot by id, name or alias and acts in every match (#784
    review); a slot nothing matches is shown as given.
    """
    shown = _slot_line(value)
    if not shown:
        return ""
    return _resolved_places(hass, value, kind) or shown


def _resolved_places(hass: HomeAssistant | None, value: Any, kind: str) -> str:
    """
    Return the names of the areas or floors a slot matches, as HA matches it.

    HA's intents accept an id, a name or an alias and act in every match, so
    a slot "garage" may be an area renamed "Guest Room" (#784 review). "" when
    nothing matches or the registries cannot be read: the slot is shown.
    """
    if (
        hass is None
        or value is None
        or isinstance(value, (list, dict))
        or getattr(hass, "data", None) is None
    ):
        return ""
    # HA reads the slot as text (`cv.string`): floor 0 is the floor "0".
    value = str(value)
    try:
        if kind == "area":
            names = [
                place.name for place in ha_intent.find_areas(value, ar.async_get(hass))
            ]
        else:
            names = [
                place.name for place in ha_intent.find_floors(value, fr.async_get(hass))
            ]
    except Exception:  # noqa: BLE001 - the slot as given still names it
        return ""
    return ", ".join(
        dict.fromkeys(_slot_line(name.translate(_QUOTE_MARKS)) for name in names)
    )


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
        return _script_line(hass, script_id, tool_args)
    # A tool named like a script (HA before 2026.9 named script tools so) is
    # held for the script or for the call its arguments make, so the line
    # names the script, and the call too when it makes one.
    bare_script = bare_script_tool_entity_id(tool_name, hass)
    # A tool may name its service by `action`, a dotted `service` or its own
    # name (#837): the line names each service the gate holds the call for.
    names = call_service_names(tool_args)
    bare_services = names[1] if names is not None else []
    named = named_services(tool_args, tool_name, hass) or []
    if bare_script is not None and not (
        tool_args.get("domain") or named or bare_services
    ):
        return _script_line(hass, bare_script, tool_args)

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

    if is_device_intent(tool_name):
        # HA's intents read the name, area and floor slots and ignore an
        # entity id and any `service` the model adds, so the line names the
        # intent and those slots (#786 review). With no name, the intent acts
        # on every device it handles.
        return pin_msg(
            hass,
            "action_service",
            service=sanitize_tool_text(
                base_tool_name(tool_name), limit=_ACTION_SERVICE_MAX_LEN
            ),
            target=_intent_target(
                hass,
                tool_args,
                _single_kind(hass, tool_name, tool_args) or _SEVERAL_KINDS,
            ),
        )
    lines = [
        _service_line(hass, {**tool_args, "domain": domain}, call)
        for domain, _, call in (name.partition(".") for name in named)
    ]
    lines += [_service_line(hass, tool_args, call) for call in bare_services]
    if not lines:
        lines.append(_service_line(hass, tool_args, base_tool_name(tool_name)))
    line = ", ".join(dict.fromkeys(lines))
    if bare_script is not None:
        return ", ".join([_script_line(hass, bare_script), line])
    return line


def _service_line(
    hass: HomeAssistant | None, tool_args: Mapping[str, Any], service: str
) -> str:
    """Name one service a tool calls, and the targets or states it sets."""
    line = _scene_state_line(hass, tool_args, service)
    if line is not None:
        return line
    return pin_msg(
        hass,
        "action_service",
        service=sanitize_tool_text(
            service.replace("_", " "), limit=_ACTION_SERVICE_MAX_LEN
        ),
        target=_service_target(hass, tool_args),
    )


def _entity_target(hass: HomeAssistant | None, entity_id: str) -> str:
    """Name one entity by its name and id, or by its id when HA has no name."""
    friendly = _entity_friendly_name(hass, entity_id)
    # An entity id is shown whole: two ids sharing a long prefix are two
    # devices (#786 review). Only its shape is allowed, so it is plain text.
    shown = _one_line(entity_id, _ENTITY_ID_MAX_LEN)
    if friendly and friendly != shown:
        return pin_msg(hass, "target_entity", name=friendly, entity_id=shown)
    return _named(hass, shown)


def _joined(hass: HomeAssistant | None, parts: list[str], *, others: bool) -> str:
    """
    Name every part, and say so when the call reaches more than it names.

    Nothing is cut: an "and N more" could hide the unlock (#786 review).
    """
    if not parts:
        return pin_msg(
            hass, "target_unnamed_devices" if others else "action_unnamed_target"
        )
    text = ", ".join(parts)
    return pin_msg(hass, "target_and_others", targets=text) if others else text


_ENTITY_ID_MAX_LEN = 255
_SCENE_ID_SHAPE = re.compile(r"^[a-z0-9_]{1,80}$")
_ENTITY_ID_SHAPE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_REGISTRY_ID_SHAPE = re.compile(r"^[0-9a-f]{32}$")
# The states a scene can set that the line names; anything else (a value HA
# rejects, or text the model wrote) reads as "change" (#786 review).
_KNOWN_STATES = frozenset(
    {
        "on",
        "off",
        "open",
        "closed",
        "locked",
        "unlocked",
        "armed_away",
        "armed_home",
        "armed_night",
        "armed_vacation",
        "armed_custom_bypass",
        "disarmed",
        "home",
        "not_home",
        "playing",
        "paused",
        "idle",
        "heat",
        "cool",
        "heat_cool",
        "auto",
        "dry",
        "fan_only",
    }
)
_EXPANDING_DOMAINS = frozenset({"group"})
_NON_ENTITY_TARGETS = ("area_id", "device_id", "floor_id", "label_id")


def _entity_ids_in(hass: HomeAssistant | None, value: Any) -> tuple[list[str], bool]:
    """
    Return the entity ids a target value names, and if it names anything else.

    Comma-joined ids are split and registry ids resolved, as HA does; `all`,
    a group, a template or an unknown id is "something else".
    """
    values = value if isinstance(value, (list, tuple)) else [value]
    found: list[str] = []
    others = False
    for item in values:
        if item is None:
            continue
        if not isinstance(item, str):
            others = True
            continue
        for part in item.split(","):
            entity_id = part.strip().lower()
            if not entity_id:
                continue
            if _REGISTRY_ID_SHAPE.fullmatch(entity_id) and hass is not None:
                try:
                    resolved = er.async_resolve_entity_id(er.async_get(hass), entity_id)
                except Exception:  # noqa: BLE001 - unnamed, so marked as others
                    resolved = None
                entity_id = (resolved or "").lower()
            if not _ENTITY_ID_SHAPE.fullmatch(entity_id):
                others = True
                continue
            if entity_id.split(".", 1)[0] in _EXPANDING_DOMAINS:
                others = True
            found.append(entity_id)
    return found, others


def _call_targets(
    hass: HomeAssistant | None, tool_args: Mapping[str, Any]
) -> tuple[list[str], bool]:
    """
    Return the entities a service call names anywhere, and if it reaches more.

    Read at the top, under `target` and under `data` (#786), as the PIN check
    reads them; an area, device, floor or label anywhere means it reaches
    devices it does not name (#786 review).
    """
    sources: list[Mapping[str, Any]] = [tool_args] + [
        tool_args[key]
        for key in ("target", *DATA_CONTAINER_KEYS)
        if isinstance(tool_args.get(key), Mapping)
    ]
    found: list[str] = []
    others = any(
        tool_args.get(key) is not None and not isinstance(tool_args.get(key), Mapping)
        for key in ("target", *DATA_CONTAINER_KEYS)
    )
    for source in sources:
        for key in ("entity_id", "entity_ids"):
            names, more = _entity_ids_in(hass, source.get(key))
            found += names
            others = others or more
        others = others or any(source.get(key) for key in _NON_ENTITY_TARGETS)
    return list(dict.fromkeys(found)), others


def _service_target(hass: HomeAssistant | None, tool_args: Mapping[str, Any]) -> str:
    """Name what a service call acts on: every entity it names, and if more."""
    found, others = _call_targets(hass, tool_args)
    if not found and isinstance(tool_args.get("name"), str):
        name = _one_line(tool_args.get("name"), _ACTION_TARGET_MAX_LEN)
        if name:
            return _joined(hass, [_named(hass, name)], others=others)
    return _joined(
        hass, [_entity_target(hass, entity_id) for entity_id in found], others=others
    )


def _scene_fields(tool_args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a scene call's state fields from the top and `data`, or None."""
    fields: dict[str, Any] = {
        key: tool_args[key] for key in INLINE_STATE_KEYS if key in tool_args
    }
    for container in DATA_CONTAINER_KEYS:
        block = tool_args.get(container)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            return None
        for key in INLINE_STATE_KEYS:
            if key in block:
                if key in fields:
                    return None
                fields[key] = block[key]
    return fields


def _scene_entries(
    tool_args: Mapping[str, Any],
) -> list[tuple[str, Any, bool]] | None:
    """
    Return (entity id, state, has state) for each state a scene call sets.

    Read from the top and from `data`, as a service tool may pass either. Two
    copies of a field, or a field HA would not read, give None: the line does
    not pick one copy over the other (#786 review).
    """
    fields = _scene_fields(tool_args)
    if fields is None:
        return None
    entries: list[tuple[str, Any, bool]] = []
    entities = fields.get("entities")
    if isinstance(entities, Mapping):
        for entity_id, value in entities.items():
            state = value.get("state") if isinstance(value, Mapping) else value
            if isinstance(value, Mapping) and set(value) - {"state"}:
                # Attributes can decide the effect (a cover's position wins
                # over its state, #786 review): the state is not shown.
                state = None
            entries.append((str(entity_id), state, True))
    elif entities is not None:
        return None
    snapshot = fields.get("snapshot_entities")
    if isinstance(snapshot, str):
        snapshot = [snapshot]
    if isinstance(snapshot, (list, tuple)):
        entries += [(str(entity_id), None, False) for entity_id in snapshot]
    elif snapshot is not None:
        return None
    return entries


def _scene_ids(tool_args: Mapping[str, Any]) -> list[Any]:
    """Return every `scene_id` a scene call gives, at the top or under `data`."""
    blocks: list[Mapping[str, Any]] = [tool_args]
    for container in DATA_CONTAINER_KEYS:
        block = tool_args.get(container)
        if isinstance(block, Mapping):
            blocks.append(block)
    return [
        block.get("scene_id") for block in blocks if block.get("scene_id") is not None
    ]


def _state_text(state: Any) -> str:
    """Return a state as HA reads it, or "" when it is not one to show."""
    if isinstance(state, bool):
        return "on" if state else "off"
    if isinstance(state, str) and state in _KNOWN_STATES:
        return state
    return ""


def _scene_state_line(
    hass: HomeAssistant | None, tool_args: Mapping[str, Any], service: str
) -> str | None:
    """
    Name the states a service tool's `scene.apply` or `scene.create` sets.

    Every state is named (#786): "set "Front Door" (lock.front_door) to
    unlocked". A value HA would not read as a plain state is not shown. A
    created scene is named with what it stores: creating it changes nothing
    until it is turned on. None when the call is not one, or its states
    cannot be read.
    """
    raw_domains = tool_args.get("domain")
    domains = raw_domains if isinstance(raw_domains, list) else [raw_domains]
    if (
        _SCENE_DOMAIN not in {str(domain).lower() for domain in domains}
        or service.lower() not in _SCENE_INLINE_SERVICES
    ):
        return None
    entries = _scene_entries(tool_args)
    if not entries:
        # States it sets that the line cannot read still run.
        return pin_msg(
            hass,
            "action_service",
            service=sanitize_tool_text(service, limit=_ACTION_SERVICE_MAX_LEN),
            target=pin_msg(hass, "target_unnamed_devices"),
        )
    creating = service.lower() == "create"
    parts: list[str] = []
    others = False
    for raw_id, state, has_state in entries:
        # A scene key is one entity id, as HA validates it: not a list, not
        # a registry id (#786 review).
        key = raw_id.strip().lower()
        found = [key] if _ENTITY_ID_SHAPE.fullmatch(key) else []
        others = others or not found
        for entity_id in found:
            target = _entity_target(hass, entity_id)
            text = _state_text(state) if has_state else ""
            if creating:
                parts.append(
                    pin_msg(hass, "target_to_state", target=target, state=text)
                    if text
                    else target
                )
            elif text:
                parts.append(
                    pin_msg(hass, "action_set_state", target=target, state=text)
                )
            else:
                parts.append(pin_msg(hass, "action_change", target=target))
    states = _joined(hass, parts, others=others)
    if not creating:
        return states
    given = _scene_ids(tool_args)
    # Only a slug HA accepts as a scene id is shown: anything else is text
    # the model wrote, and HA rejects it (#786 review).
    scene = (
        given[0]
        if len(given) == 1
        and isinstance(given[0], str)
        and _SCENE_ID_SHAPE.fullmatch(given[0])
        else ""
    )
    return pin_msg(
        hass,
        "action_create_scene",
        scene=_named(hass, scene) if scene else pin_msg(hass, "action_unnamed_target"),
        states=states,
    )


def _script_line(
    hass: HomeAssistant | None,
    script_id: str,
    tool_args: Mapping[str, Any] | None = None,
) -> str:
    """
    Name running a script, from Home Assistant, and the devices passed to it.

    A script's arguments are free-form fields the model fills in, so the line
    names the script from Home Assistant, and adds only what a field that
    selects an entity, area, device, floor, label or target resolves to in
    Home Assistant's registries (#787): text the model writes cannot change
    the line, but the door it picked is shown.
    """
    state = hass.states.get(script_id) if hass is not None else None
    name = state.name if state is not None else script_id
    line = pin_msg(
        hass,
        "action_turn_on",
        target=_named(hass, _one_line(name, _ACTION_TARGET_MAX_LEN)),
    )
    if hass is None or not tool_args:
        return line
    # The arguments as they are sent: the normal path and the confirmation
    # both reduce them first (`minimal_payload_for_domain`).
    sent = minimal_payload_for_domain(dict(tool_args))
    try:
        names, others = _script_field_targets(hass, script_id, sent)
    except Exception:
        # The names only add to the line: they must never cost the prompt.
        LOGGER.exception("Critical Action PIN: could not name a script's targets")
        return line
    if not names and not others:
        return line
    return pin_msg(
        hass,
        "action_script_targets",
        action=line,
        targets=_joined(hass, names, others=others),
    )


# Selector types whose value names something in Home Assistant's registries.
_TARGET_SELECTORS = ("target", "entity", "device", "area", "floor", "label")
_TARGET_KEYS = {
    "entity_id": "entity",
    "area_id": "area",
    "device_id": "device",
    "floor_id": "floor",
    "label_id": "label",
}


def _script_field_targets(
    hass: HomeAssistant, script_id: str, tool_args: Mapping[str, Any]
) -> tuple[list[str], bool]:
    """
    Return the names of what a script's selector fields were given.

    And whether a field holds something that could not be named (#787
    review): a comma-joined list, an unknown id, a value of the wrong shape.
    Those still reach the script, so the line says it reaches other devices.
    """
    # HA's script tool calls the service its registry entry's unique id
    # names, which a renamed script keeps (#787 review).
    service_name = script_id.split(".", 1)[1]
    entry = er.async_get(hass).async_get(script_id)
    if entry is not None and entry.unique_id:
        service_name = entry.unique_id
    description = async_get_cached_service_description(hass, "script", service_name)
    fields = description.get("fields") if isinstance(description, Mapping) else None
    if not isinstance(fields, Mapping):
        return [], False
    names: list[str] = []
    others = False
    for field, config in sorted(fields.items()):
        selector = config.get("selector") if isinstance(config, Mapping) else None
        if not isinstance(selector, Mapping) or field not in tool_args:
            continue
        kind = next((k for k in _TARGET_SELECTORS if k in selector), None)
        if kind is None:
            continue
        options = selector.get(kind)
        multiple = isinstance(options, Mapping) and bool(options.get("multiple"))
        value = tool_args[field]
        if isinstance(value, Mapping):
            # A target, or an entity field given as one.
            # HA passes a target on as it is, so ids are looked up exactly,
            # not by name (#787 review).
            for key, target_kind in _TARGET_KEYS.items():
                found, more = _registry_names(
                    hass, target_kind, value.get(key), multiple=True, exact=True
                )
                names += found
                others = others or more
            continue
        found, more = _registry_names(hass, kind, value, multiple=multiple)
        names += found
        others = others or more
    return list(dict.fromkeys(names)), others


def _registry_names(
    hass: HomeAssistant,
    kind: str,
    value: Any,
    *,
    multiple: bool,
    exact: bool = False,
) -> tuple[list[str], bool]:
    """
    Return the quoted names a value resolves to, and if some did not resolve.

    Resolved as Home Assistant's script tool resolves them: an area or floor
    by id, name or alias (only the first match for a single selector), an
    entity by id or registry id.
    """
    if value is None:
        return [], False
    if kind == "entity":
        entity_ids, others = _entity_ids_in(hass, value)
        names: list[str] = []
        for entity_id in entity_ids:
            if hass.states.get(entity_id) is None:
                others = True
                continue
            # Named with its id: two devices of one name stay two (#787 review).
            names.append(_entity_target(hass, entity_id))
        return names, others
    values = value if isinstance(value, (list, tuple)) else [value]
    names = []
    others = False
    for item in values:
        found = _registry_item_names(hass, kind, item, exact=exact)
        if not found:
            others = True
            continue
        names += found if multiple else found[:1]
    return (
        [_named(hass, _slot_line(name.translate(_QUOTE_MARKS))) for name in names],
        others,
    )


def _registry_item_names(
    hass: HomeAssistant, kind: str, item: Any, *, exact: bool = False
) -> list[str]:
    """Return the names one area, floor, device or label value matches."""
    if not isinstance(item, str):
        return []
    if kind in {"area", "floor"}:
        return _place_names(hass, kind, item, exact=exact)
    if kind == "device":
        device = dr.async_get(hass).async_get(item)
        name = (device.name_by_user or device.name) if device else None
    elif kind == "label":
        labels = lr.async_get(hass)
        label = labels.async_get_label(item) or (
            None if exact else labels.async_get_label_by_name(item)
        )
        name = label.name if label else None
    else:
        name = None
    return [name] if name else []


def _place_names(
    hass: HomeAssistant, kind: str, item: str, *, exact: bool
) -> list[str]:
    """Return the areas or floors a value names: by id, or as HA matches it."""
    if kind == "area":
        areas = ar.async_get(hass)
        if exact:
            area = areas.async_get_area(item)
            return [area.name] if area else []
        return [area.name for area in ha_intent.find_areas(item, areas)]
    floors = fr.async_get(hass)
    if exact:
        floor = floors.async_get_floor(item)
        return [floor.name] if floor else []
    return [floor.name for floor in ha_intent.find_floors(item, floors)]

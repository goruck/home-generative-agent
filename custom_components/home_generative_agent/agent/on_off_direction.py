"""
Keep HassTurnOn / HassTurnOff going the way the user asked (#797).

For a cover, HassTurnOn opens and HassTurnOff closes; for a lock, HassTurnOn
locks and HassTurnOff unlocks. Both are bound together, and a small model can
call the wrong one: "open the study shade" ran HassTurnOff and closed it.
Home Assistant's result names the entity but not what was done, so the
model's reply then described what it meant to do ("the shade is now open").

* **Before the call** (``wrong_direction_refusal``): a short request is split
  into phrases (at punctuation and at "and", "then", "but", "also"). When
  exactly one phrase gives a command, and that command is ``open``,
  ``close``, ``shut``, ``lock``, ``unlock``, ``press`` or ``turn``/``switch``
  /``power`` ``on``/``off`` at the start of the phrase ("Open all shades",
  "please close it", "would you like me to open it"), a call going the other
  way is not run; the model is told which tool does what was asked.

  Left alone, since the command may be another target's: a request with a
  second commanding phrase ("lights off and open the blinds", "close the
  shade and start the coffee maker"), a phrase whose bare ``on``/``off``
  disagrees with its command ("open plan lights off"), a long request, one
  in another language, and a call on a script, scene, automation or button
  (run, not turned). A direction word that is not at the start of a phrase
  is a state or a name ("the garage is open") and is not read. The call's
  own target names (name, area, floor, entity id and the entity's friendly
  name) are taken out of the request first, unless a name is a command word
  itself ("Lock"). Open and close are read only for a target known not to
  be a lock ("open the front door lock" unlocks it).
* **After the call** (``with_action_performed``): the result says which
  command was sent to each entity ("close command sent to Study Shade"),
  whatever the request, so the reply can match it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .helpers import base_tool_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

ON = "on"
OFF = "off"
_TOOL_DIRECTIONS = {"HassTurnOn": ON, "HassTurnOff": OFF}
_TOOL_FOR = {ON: "HassTurnOn", OFF: "HassTurnOff"}

# A request's phrases: one command each, at most.
_PHRASE_BREAK = re.compile(r"[.!?;,:]+|\b(?:and|then|but|plus|also)\b", re.IGNORECASE)
# Where a phrase's command sits: at its start (after filler), or after a
# polite or offering opener.
_COMMAND_START = (
    r"(?:^\s*(?:(?:please|kindly|now|just|ok|okay|yes|yeah|hey|so)\s+)*"
    r"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"|\bgo\s+ahead\s+and\s+|\bme\s+to\s+|\b(?:should|shall|can|may)\s+i\s+"
    r"|\bi\s+(?:can|could|will)\s+|\bi'll\s+|\blet\s+me\s+)"
)
# A command, by kind: a cover or valve moved, a lock set, anything switched.
_MOVE, _LOCK, _SWITCH = "move", "lock", "switch"
_COMMAND = re.compile(
    _COMMAND_START + r"(?:(open|press)|(close|shut)|(lock)|(unlock)"
    r"|(?:turn|switch|power)\s+(?:[\w']+\s+){0,4}?(on|off))\b",
    re.IGNORECASE,
)
_SWITCH_VERB = re.compile(r"\b(?:turn|switch|power)\b", re.IGNORECASE)
_BARE = re.compile(r"\b(on|off)\b", re.IGNORECASE)
# Another command: a phrase giving one is a second command.
_OTHER_ACTION = re.compile(
    _COMMAND_START
    + r"(?:start|stop|set|run|play|pause|resume|dim|brighten|mute|unmute|"
    r"trigger|toggle|arm|disarm|activate|deactivate|enable|disable|raise|"
    r"lower|increase|decrease|skip|kill|cut)\b",
    re.IGNORECASE,
)
# Command words a target's name is not taken out as ("the Lock").
_COMMAND_WORDS = frozenset(
    {"open", "close", "shut", "lock", "unlock", "press", "on", "off"}
)
# Longer than a spoken command: not read.
_MAX_REQUEST_CHARS = 300
# Calls that run something rather than turn it one way, and the domains
# whose HassTurnOn is "run".
_NOT_CHECKED_DOMAINS = frozenset(
    {"script", "scene", "automation", "button", "input_button"}
)
_LOCK_DOMAIN = "lock"
# Where a call names its target: these words are a name, not a command.
_NAME_KEYS = ("name", "area", "floor", "entity_id")
_SEPARATORS = re.compile(r"[-_]+")
_ENTITY_ID = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")
# A reply that only accepts the agent's offer, in the UI languages: every
# word one of these (a word list, not a regex, so no input backtracks).
_YES_WORDS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "alright",
        "please",
        "absolutely",
        "definitely",
        "go",
        "ahead",
        "do",
        "it",
        "ano",
        "jo",
        "jasně",
        "да",
        "давай",
        "конечно",
        "evet",
        "tamam",
    }
)
# "go", "do" and "it" only as "go ahead" and "do it".
_YES_PHRASE_WORDS = frozenset({"go", "ahead", "do", "it"})
_YES_PHRASES = (("go", "ahead"), ("do", "it"))
_WORD = re.compile(r"\w+")
_MAX_YES_WORDS = 6

# What each tool does, by domain (Home Assistant's OnOffIntentHandler).
# The command, not its outcome: HA reports the call accepted, while a cover
# may still be moving or a lock jam.
_DONE: dict[str, tuple[str, str]] = {
    "cover": ("open", "close"),
    "valve": ("open", "close"),
    "lock": ("lock", "unlock"),
    "button": ("press", "turn off"),
    "input_button": ("press", "turn off"),
}
_DEFAULT_DONE = ("turn on", "turn off")
_REQUEST_WORDS = {ON: "open / turn on / lock", OFF: "close / turn off / unlock"}


def tool_direction(tool_name: str) -> str | None:
    """Return ``on`` for HassTurnOn, ``off`` for HassTurnOff, else None."""
    return _TOOL_DIRECTIONS.get(base_tool_name(tool_name))


def is_bare_yes(text: str) -> bool:
    """Return True for a reply that only accepts an offer ("Yes", "Sure, go ahead")."""
    words = _WORD.findall(text.lower())
    if not words or len(words) > _MAX_YES_WORDS:
        return False
    rest = list(words)
    for phrase in _YES_PHRASES:
        for index in range(len(rest) - 1):
            if tuple(rest[index : index + 2]) == phrase:
                rest[index : index + 2] = ["yes"]
                break
    return all(w in _YES_WORDS and w not in _YES_PHRASE_WORDS for w in rest)


def requested_direction(text: str) -> tuple[str, str] | None:
    """
    Return the one direction a short request commands, and its words.

    None when it commands none, more than one thing, or is long.
    """
    found = _command(text)
    return (found[0], found[1]) if found else None


def _command(text: str) -> tuple[str, str, str] | None:  # noqa: PLR0911
    """Return the request's one command: direction, words and kind."""
    text = _SEPARATORS.sub(" ", text.replace("\u2019", "'")).strip()
    if not text or len(text) > _MAX_REQUEST_CHARS:
        return None
    commanding = [
        phrase
        for phrase in _PHRASE_BREAK.split(text)
        if _COMMAND.search(phrase)
        or _OTHER_ACTION.search(phrase)
        or _BARE.search(phrase)
    ]
    if len(commanding) != 1:
        return None
    phrase = commanding[0]
    if _OTHER_ACTION.search(phrase):
        return None
    matches = list(_COMMAND.finditer(phrase))
    if len(matches) != 1:
        return None
    match = matches[0]
    move_on, move_off, lock, unlock, split = match.group(1, 2, 3, 4, 5)
    if split:
        direction, words, kind = split.lower(), match.group(0).strip(), _SWITCH
    elif move_on or move_off:
        direction, words, kind = ON if move_on else OFF, move_on or move_off, _MOVE
    else:
        direction, words, kind = ON if lock else OFF, lock or unlock, _LOCK
    if len(_SWITCH_VERB.findall(phrase)) > (1 if split else 0):
        return None  # a "turn" whose on or off was not read
    rest = phrase[: match.start()] + " " + phrase[match.end() :]
    if any(bare.lower() != direction for bare in _BARE.findall(rest)):
        return None  # "open plan lights off": the open may be a name
    return direction, words, kind


def wrong_direction_refusal(
    tool_name: str,
    tool_args: Mapping[str, Any],
    request_text: str,
    *,
    known: tuple[Iterable[str], Iterable[str]] = ((), ()),
) -> str | None:
    """
    Return why the call is not run, when it goes against the request.

    None when the tool is not HassTurnOn/HassTurnOff, the call is not one
    this checks, the request does not give one command, or the call goes
    the way it does (see the module docstring). ``known`` adds the target's
    domains and names Home Assistant knows (from its entity id or name),
    when the call does not give them.
    """
    called = tool_direction(tool_name)
    if called is None:
        return None
    domains, names = _target(tool_args)
    domains |= {str(d).lower() for d in known[0]}
    names |= {_SEPARATORS.sub(" ", str(n)).strip() for n in known[1] if str(n).strip()}
    if domains & _NOT_CHECKED_DOMAINS:
        return None
    text = _SEPARATORS.sub(" ", request_text)
    for name in sorted(names, key=len, reverse=True):
        if name.lower() in _COMMAND_WORDS:
            continue  # "Lock the lock": the command stays
        text = re.sub(rf"\b{re.escape(name)}\b", " ", text, flags=re.IGNORECASE)
    asked = _command(text)
    if asked is None or asked[0] == called:
        return None
    direction, word, kind = asked
    is_lock = _LOCK_DOMAIN in domains
    # "Open the front door lock" unlocks it: open and close are read only
    # for a target known not to be a lock, and on and off not for a lock.
    if (kind == _MOVE and (not domains or is_lock)) or (
        kind == _LOCK and domains and not is_lock
    ):
        return None
    if kind == _SWITCH and is_lock:
        return None
    right = tool_name.removesuffix(base_tool_name(tool_name)) + _TOOL_FOR[direction]
    return (
        f"Not run: the user said '{word}', and {tool_name} does the opposite "
        f"({_REQUEST_WORDS[called]}). Call {right} with the same arguments instead."
    )


def target_entity_ids(tool_args: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return the entity ids and the other names a call gives its target."""
    ids: list[str] = []
    others: list[str] = []
    for key in ("name", "entity_id"):
        values = tool_args.get(key) or []
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, str) and value.strip():
                if _ENTITY_ID.fullmatch(value.strip().lower()):
                    ids.append(value.strip().lower())
                elif key == "name":
                    others.append(value.strip())
    return ids, others


def _target(tool_args: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return the call's target domains and the names it gives its target."""
    raw = tool_args.get("domain") or []
    domains = {str(d).lower() for d in (raw if isinstance(raw, list) else [raw])}
    names: set[str] = set()
    for key in _NAME_KEYS:
        values = tool_args.get(key) or []
        for value in values if isinstance(values, list) else [values]:
            if not isinstance(value, str) or not value.strip():
                continue
            name = value.strip()
            if _ENTITY_ID.fullmatch(name.lower()):
                # An entity id: its domain, and its object id as a name.
                domain, _, name = name.partition(".")
                domains.add(domain.lower())
            names.add(_SEPARATORS.sub(" ", name).strip())
    return domains - {""}, {name for name in names if name}


def with_action_performed(tool_name: str, data: Any) -> Any:
    """
    Return the result with what was done to each entity added.

    Only for a HassTurnOn/HassTurnOff result naming the entities it acted on;
    anything else is returned as it is.
    """
    called = tool_direction(tool_name)
    if called is None or not isinstance(data, dict):
        return data
    inner = data.get("data")
    success = inner.get("success") if isinstance(inner, dict) else None
    if not isinstance(success, list):
        return data
    done: list[str] = []
    for target in success:
        if not isinstance(target, dict):
            continue
        entity_id = str(target.get("id") or "")
        if target.get("type") != "entity" or "." not in entity_id:
            continue
        verbs = _DONE.get(entity_id.split(".", 1)[0], _DEFAULT_DONE)
        verb = verbs[0] if called == ON else verbs[1]
        done.append(f"{verb} command sent to {target.get('name') or entity_id}")
    if not done:
        return data
    return {**data, "action_performed": done}

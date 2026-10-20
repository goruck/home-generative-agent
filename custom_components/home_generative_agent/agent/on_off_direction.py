"""
Keep HassTurnOn / HassTurnOff going the way the user asked (#797).

For a cover, HassTurnOn opens and HassTurnOff closes; for a lock, HassTurnOn
locks and HassTurnOff unlocks. Both are bound together, and a small model can
call the wrong one: "open the study shade" ran HassTurnOff and closed it.
Home Assistant's result names the entity but not what was done, so the
model's reply then described what it meant to do ("the shade is now open").

* **Before the call** (``wrong_direction_refusal``): when a short request
  gives one command, ``open``, ``close``, ``shut``, ``lock``, ``unlock``,
  ``press`` or ``turn``/``switch``/``power`` ``on``/``off``, at the start of a
  clause ("Open all shades", "please close it", "can you turn the fan on"),
  and the tool goes the other way, it is not run; the model is told which
  tool does what was asked. A direction word anywhere else is a state or a
  name ("the garage is open", "the open plan lights") and is not read. Left
  alone: a request with both directions or another command ("close the
  shade and start the coffee maker", "turn it off and on again"), a long
  one, one in another language, and a call on a script, scene, automation
  or button (run, not turned). The call's own target names (its name, area,
  floor or entity id) are taken out of the request first, so "kill the open
  plan lights" reads no "open". Open and close are read only for a target
  known not to be a lock ("open the front door lock" unlocks it).
* **After the call** (``with_action_performed``): the result says which
  command was sent to each entity ("close command sent to Study Shade"),
  whatever the request, so the reply can match it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .helpers import base_tool_name

if TYPE_CHECKING:
    from collections.abc import Mapping

ON = "on"
OFF = "off"
_TOOL_DIRECTIONS = {"HassTurnOn": ON, "HassTurnOff": OFF}
_TOOL_FOR = {ON: "HassTurnOn", OFF: "HassTurnOff"}

# Where a command starts: the text, a clause, or after a polite or
# connecting word. A direction word elsewhere is a state or part of a name.
_CLAUSE_START = (
    r"(?:^|[.!?;,:]\s*|\b(?:and|then|also|now|just|please|kindly|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|go\s+ahead\s+and|"
    # The agent's offer: "Would you like me to open it?", "Should I close
    # it?", "I can open the shade."
    r"me\s+to|(?:should|shall|can|may)\s+i|i\s+(?:can|could|will)|i'll|"
    r"let\s+me)\s+)"
)
# A command, by kind: a cover or valve moved, a lock set, anything switched.
_MOVE, _LOCK, _SWITCH = "move", "lock", "switch"
_COMMAND = re.compile(
    _CLAUSE_START + r"(?:(open|press)|(close|shut)|(lock)|(unlock)"
    r"|(?:turn|switch|power)\s+(?:[\w']+\s+){0,4}?(on|off))\b",
    re.IGNORECASE,
)
_SWITCH_VERB = re.compile(r"\b(?:turn|switch|power)\b", re.IGNORECASE)
_BARE_ON = re.compile(r"\bon\b", re.IGNORECASE)
_BARE_OFF = re.compile(r"\boff\b", re.IGNORECASE)
# Another command in the same request: its call is not this one's.
_OTHER_ACTION = re.compile(
    _CLAUSE_START + r"(?:start|stop|set|run|play|pause|resume|dim|brighten|mute|unmute|"
    r"trigger|toggle|arm|disarm|activate|deactivate|enable|disable|raise|"
    r"lower|increase|decrease|skip)\b",
    re.IGNORECASE,
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


def requested_direction(text: str) -> tuple[str, str] | None:
    """
    Return the one direction a short request commands, and its words.

    None when it commands none, both, or another action too, or is long.
    """
    found = _command(text)
    return (found[0], found[1]) if found else None


def _command(text: str) -> tuple[str, str, str] | None:
    """Return the request's one command: direction, words and kind."""
    text = _SEPARATORS.sub(" ", text.replace("\u2019", "'")).strip()
    if not text or len(text) > _MAX_REQUEST_CHARS or _OTHER_ACTION.search(text):
        return None
    if _BARE_ON.search(text) and _BARE_OFF.search(text):
        return None  # "off and on again", "the light on the porch off"
    found: dict[str, tuple[str, str]] = {}
    splits = 0
    for match in _COMMAND.finditer(text):
        move_on, move_off, lock, unlock, split = match.group(1, 2, 3, 4, 5)
        if move_on or move_off:
            found.setdefault(ON if move_on else OFF, (move_on or move_off, _MOVE))
        elif lock or unlock:
            found.setdefault(ON if lock else OFF, (lock or unlock, _LOCK))
        else:
            splits += 1
            words = match.group(0).strip(" .!?;,:")
            found.setdefault(split.lower(), (words, _SWITCH))
    if splits < len(_SWITCH_VERB.findall(text)):
        return None  # a "turn" whose on or off was not read
    if len(found) != 1:
        return None
    direction, (words, kind) = next(iter(found.items()))
    return direction, words, kind


def wrong_direction_refusal(
    tool_name: str, tool_args: Mapping[str, Any], request_text: str
) -> str | None:
    """
    Return why the call is not run, when it goes against the request.

    None when the tool is not HassTurnOn/HassTurnOff, the call is not one
    this checks (see the module docstring), the request does not command one
    direction, or the call goes the way it does. The target's own names are
    taken out of the request first: in "kill the open plan lights", "open"
    is the lights' name.
    """
    called = tool_direction(tool_name)
    if called is None:
        return None
    domains, names = _target(tool_args)
    if domains & _NOT_CHECKED_DOMAINS:
        return None
    text = _SEPARATORS.sub(" ", request_text)
    for name in sorted(names, key=len, reverse=True):
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
    return domains, {name for name in names if name}


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

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
  alone: a request with both directions or another action verb ("close the
  shade and start the coffee maker"), a long one, one in another language, a
  call on a script, scene, automation or button (run, not turned), and a
  target whose name holds a direction word ("Open Garage").
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
    # The agent's offer: "Would you like me to open it?", "Should I close it?"
    r"me\s+to|(?:should|shall|can|may)\s+i)\s+)"
)
_COMMAND = re.compile(
    _CLAUSE_START + r"(?:(open|lock|press)|(close|shut|unlock)"
    r"|(?:turn|switch|power)\s+(?:[\w'-]+\s+){0,4}?(on|off))\b",
    re.IGNORECASE,
)
# Another action in the same request: its call is not this command's.
_OTHER_ACTION = re.compile(
    r"\b(?:start|stop|set|run|play|pause|resume|dim|brighten|mute|unmute|"
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
    text = text.replace("\u2019", "'").strip()
    if not text or len(text) > _MAX_REQUEST_CHARS or _OTHER_ACTION.search(text):
        return None
    found: dict[str, str] = {}
    for match in _COMMAND.finditer(text):
        on, off, split = match.group(1, 2, 3)
        if on:
            found.setdefault(ON, on)
        elif off:
            found.setdefault(OFF, off)
        else:
            found.setdefault(split.lower(), match.group(0).strip(" .!?;,:"))
    if len(found) != 1:
        return None
    direction, word = next(iter(found.items()))
    return direction, word


def wrong_direction_refusal(
    tool_name: str, tool_args: Mapping[str, Any], request_text: str
) -> str | None:
    """
    Return why the call is not run, when it goes against the request.

    None when the tool is not HassTurnOn/HassTurnOff, the call is not one
    this checks (see the module docstring), the request does not command one
    direction, or the call goes the way it does.
    """
    called = tool_direction(tool_name)
    if called is None or _not_checked(tool_args):
        return None
    asked = requested_direction(request_text)
    if asked is None or asked[0] == called:
        return None
    direction, word = asked
    right = tool_name.removesuffix(base_tool_name(tool_name)) + _TOOL_FOR[direction]
    return (
        f"Not run: the user said '{word}', and {tool_name} does the opposite "
        f"({_REQUEST_WORDS[called]}). Call {right} with the same arguments instead."
    )


def _not_checked(tool_args: Mapping[str, Any]) -> bool:
    """Return True for a script, scene or button, or a name with a direction."""
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    if {str(d).lower() for d in domains} & _NOT_CHECKED_DOMAINS:
        return True
    names = tool_args.get("name") or []
    names = names if isinstance(names, list) else [names]
    return any(
        _NAME_DIRECTION.search(str(name)) for name in names if isinstance(name, str)
    )


_NAME_DIRECTION = re.compile(
    r"\b(?:open|close|shut|lock|unlock|press|on|off)\b", re.IGNORECASE
)


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
